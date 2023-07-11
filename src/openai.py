"""# LLM Module

Implements an interface for prompting LLMs.

Right now we are tightly coupled to OpenAI's LLMs & API but we expect to support locally running OpenSource models via GGML in the near future.
"""

import asyncio
import aiohttp
import logging
import tiktoken
import itertools
import json
import numpy as np
import time

from dataclasses import dataclass, field
from typing import Any

from ._interfaces import ChatMessage, LLM, Tokenizer, Tokens, Embeddings

logger = logging.getLogger(__name__)

OPENAI_AVAILABLE_MODELS = {
  "gpt4": {
    "kind": "chat",
    "id": "gpt-4-0613",
    "context_window": 8192,
  },
  "gpt4-32k": {
    "kind": "chat",
    "id": "gpt-4-32k-0613",
    "context_window": 32768,
  },
  "gpt3": {
    "kind": "chat",
    "id": "gpt-3.5-turbo-0613",
    "context_window": 4096,
  },
  "gpt3-16k": {
    "kind": "chat",
    "id": "gpt-3.5-turbo-16k-0613",
    "context_window": 16384,
  },
  "ada": {
    "kind": "embedding",
    "id": "text-embedding-ada-0002",
    "context_window": 8191,
    "vector_dimensions": 1536,
  }
}

OPENAI_MODEL_TUNING_PRESETS = {
  "creative": {
    "description": "Responses are more creative; the model is more curious.",
    "tuning": {
      "temperature": 2.0,
      "top_p": 0.95,
    }
  },
  "balanced": {
    "description": "Responses are more balanced; the model is more balanced.",
    "tuning": {
      "temperature": 1.0,
      "top_p": 0.815,
    }
  },
  "reserved": {
    "description": "Responses are more reserved; the model is more straightforward.",
    "tuning": {
      "temperature": 0.5,
      "top_p": 0.68,
    }
  },
}

_gpt_3_model_ids: set[str] = {
  "gpt-3.5-turbo",
  "gpt-3.5-turbo-16k",
  "gpt-3.5-turbo-0613",
  "gpt-3.5-turbo-16k-0613",
}

_gpt_4_model_ids: set[str] = {
  "gpt-4",
  "gpt-4-32k",
  "gpt-4-0613",
  "gpt-4-32k-0613",
}

class Error(Exception):
  def __init__(self, exc: Exception | None, message: str):
    super().__init__(message)
    self.exc = exc

class ModelError(Error):
  ...

class HTTPError(Error):
  def __init__(self, exc: Exception | None, status: int, headers: dict[str, Any], message: str):
    super().__init__(exc=exc, message=message)
    self.status = status
    self.message = message
    self.headers = headers
  
  def parse_rate_limit_sleep_time(self) -> float:
    """How long to wait before retrying the request"""
    # Time comes accross as a string with a unit suffix of s or ms
    assert self.status == 429, "Can only parse rate limit sleep time for 429 responses"
    wait_times = []
    for wait_time in [
      self.headers.get("x-ratelimit-reset-requests", "0s"),
      self.headers.get("x-ratelimit-reset-tokens", "0s"),
    ]:
      if wait_time.endswith("ms"):
        wait_times.append(float(wait_time[:-2]) / 1000)
      elif wait_time.endswith("s"):
        wait_times.append(float(wait_time[:-1]))
    
    return max(wait_times)
  
async def _safe_raise_for_status(response: aiohttp.ClientResponse) -> None:
  try:
    response.raise_for_status()
  except aiohttp.ClientResponseError as cre:
    # Format the response JSON encodable dict
    _data = {
      "status": cre.status,
      "message": cre.message,
      "headers": dict(response.headers),
    }
    logger.info(f"HTTP Response: {_data}")
    logger.error(f"HTTP Error: {cre.status} - {cre.message}")
    raise HTTPError(None, cre.status, dict(response.headers), cre.message)

# def _calc_tokens(model: str, message: str) -> int:
#   try:
#     encoding = tiktoken.encoding_for_model(model)
#   except KeyError:
#     logger.warning(f"Model {model} not found in TikToken. Using default encoding.")
#     encoding = tiktoken.get_encoding("cl100k_base")
  
#   return len(encoding.encode(message))

@dataclass
class _OpenAITokenizerContext:
  encoding: tiktoken.Encoding
  tokenizer: Tokenizer

@dataclass
class OpenAITokenizerManager:
  model: str
  _context: _OpenAITokenizerContext | None = field(init=False)

  async def start(self) -> Tokenizer:
    # TODO: Setup Threadpool Executor for parallel encoding/decoding
    self._context = _OpenAITokenizerContext(
      encoding=tiktoken.encoding_for_model(self.model),
      tokenizer=Tokenizer(model=self.model)
    )
    for method_name in ("encode", "decode", "tokens_consumed"):
      setattr(
        self._context.tokenizer,
        method_name,
        getattr(self, method_name)
      )
    return self._context.tokenizer
  
  async def stop(self):
    self._context = None

  async def __aenter__(self) -> Tokenizer:
    return await self.start()
  
  async def __aexit__(self, exc_type, exc_value, traceback):
    await self.stop()
  
  async def encode(self, *messages: str) -> list[list[int]]:
    assert self._context is not None
    return [self._context.encoding.encode(m, disallowed_special=()) for m in messages] # Disabled disallowed_special because we want to allow any text. Ran into an error testing on OpenAI's Paper's that had the ChatML Markup in it.
  
  async def decode(self, *tokens: list[int]) -> list[str]:
    assert self._context is not None
    return [self._context.encoding.decode(t, errors='strict') for t in tokens]

  async def tokens_consumed(self, *messages: ChatMessage) -> int:
    """Calculates the number of tokens that will be consumed by the given messages. Accounts for the particulars of OpenAI's ChatML. Useful for determining the remaing context window size.

    To get the exact size of tokenized messages, sum the length of each token list returned by encode().
    """
    assert self._context is not None
    # Note that true token count is not just a simple sum; We need to account for ChatML's special tokens: https://github.com/openai/openai-python/blob/main/chatml.md
    if self.model in _gpt_3_model_ids | _gpt_4_model_ids:
      tokens_per_message = 3
      tokens_per_name = 1
    elif self.model == "gpt-3.5-turbo-0301":
      tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
      tokens_per_name = -1  # if there's a name, the role is omitted
    else:
      raise NotImplementedError(
          f"""num_tokens_from_messages() is not implemented for model {self.model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
      )

    token_offset = 0
    if self.model in _gpt_4_model_ids:
      # I notice an off by one error in the token count for gpt-4 models. 🤷‍♂️
      token_offset = 1
    
    # tokenize the content & role for each message since they contribute to total token count
    tokens = await self.encode(*list(itertools.chain.from_iterable(
      (m.content, m.role) for m in messages
    )))
    # names = [m.name for m in messages] # TODO: ChatMessage doesn't currently have a name field

    # every reply is primed with <|start|>assistant<|message|> so add 3
    return (len(messages) * tokens_per_message) + sum(len(t) for t in tokens) + 3 + token_offset # + (len(names) * tokens_per_name) # TODO: ChatMessage doesn't currently have a name field
    
@dataclass
class OpenAIChatOptions:
  temperature: float
  max_tokens: int
  top_p: float

  def to_dict(self) -> dict[str, Any]:
    return {
      "temperature": self.temperature,
      "max_tokens": self.max_tokens,
      "top_p": self.top_p,
    }

@dataclass
class OpenAIModelInfo:
  model_id: str
  model_context_window: int

@dataclass
class OpenAIEmbeddingModelInfo:
  model_id: str
  model_context_window: int
  vector_dimensions: int

@dataclass
class _OpenAISessionManagerContext:
  is_held: bool = False
  chat_interface: LLM | None = None
  chat_tokenizer_manager: OpenAITokenizerManager | None = None
  embedding_interface: Embeddings | None = None
  embedding_tokenizer_manager: OpenAITokenizerManager | None = None
  concurrent_lock: asyncio.Semaphore | None = None
  ratelimit_gate: asyncio.Event | None = None
  ratelimit_task: asyncio.Task | None = None
  """A Gate that is closed (false) when the ratelimit is exceeded. This is used to prevent concurrent requests from being made when the ratelimit is exceeded."""

@dataclass
class OpenAISessionManager:
  chat_model: OpenAIModelInfo | None = None
  """The LLM Model to use"""
  chat_opts: OpenAIChatOptions | None = None
  """Chat Completion Options"""
  embedding_model: OpenAIEmbeddingModelInfo | None = None
  """The Text Embedding Model to use; currently only text-embedding-ada-002 is supported"""
  session: aiohttp.ClientSession = field(default_factory=aiohttp.ClientSession)
  """The aiohttp ClientSession to use for requests to the OpenAI API"""
  concurrency: int = field(default=10) # Max number of concurrent requests to OpenAI API
  """The maximum number of concurrent requests to the OpenAI API"""
  _context: _OpenAISessionManagerContext = field(init=False, default_factory=_OpenAISessionManagerContext)
  
  async def __aenter__(self) -> tuple[tuple[LLM, Tokenizer] | None, tuple[Embeddings, Tokenizer] | None]:
    return await self.start()
  
  async def __aexit__(self, exc_type, exc_value, traceback):
    await self.stop()

  async def start(self) -> tuple[tuple[LLM, Tokenizer] | None, tuple[Embeddings, Tokenizer] | None]:
    logger.info("Starting OpenAI API Session...")
    # Test Authorization
    logger.debug("Testing OpenAI API Authorization...")
    model_list: list[str]
    try:
      async with self.session.get("/v1/models") as models_resp:
        await _safe_raise_for_status(models_resp)
        models_resp_json = await models_resp.json()
        assert isinstance(models_resp_json, dict) \
          and models_resp_json.keys() >= {"object", "data"} \
          and models_resp_json["object"] == "list" \
          , "Invalid response from OpenAI API"
        logger.debug(json.dumps(sorted(models_resp_json['data'], key = lambda m: m['id']), indent=2))
        model_list = [
          m['id'] for m in models_resp_json["data"]
          if m["object"] == "model"
        ]
        assert len(model_list) > 0, "No models found"
    except Error as e:
      logger.error(f"Failed to start Chat Session: {e}")
      await self.session.close()
      raise

    assert self.chat_model is not None or self.embedding_model is not None, "Either model or embedding_model must be specified"
    
    if self.chat_model is not None:
      assert self.chat_opts is not None, "opts must be specified if model is specified"
      if self.chat_model.model_id not in model_list:
        logger.error(f"Couldn't find the requested model: {self.chat_model.model_id}")
        logger.info(f"Available models: {model_list}")
        raise ValueError(f"Couldn't find the requested model: {self.chat_model.model_id}")

      self._context.chat_interface = LLM(
        name=self.chat_model.model_id,
        context_window=self.chat_model.model_context_window,
      )
      async def _chat(messages: list[ChatMessage], **kwargs) -> ChatMessage | tuple[ChatMessage, ...]:
        return await self.chat(messages, **kwargs)
      setattr(
        self._context.chat_interface,
        "chat",
        _chat
      )
      self._context.chat_tokenizer_manager = OpenAITokenizerManager(self.chat_model.model_id)
      _chat_tokenizer_interface = await self._context.chat_tokenizer_manager.start()
    
    if self.embedding_model is not None:
      if self.embedding_model.model_id not in model_list:
        logger.error(f"Couldn't find the requested model: {self.embedding_model.model_id}")
        logger.info(f"Available models: {model_list}")
        raise ValueError(f"Couldn't find the requested model: {self.embedding_model.model_id}")

      self._context.embedding_interface = Embeddings(
        name=self.embedding_model.model_id,
        context_window=self.embedding_model.model_context_window,
        vector_dimensions=self.embedding_model.vector_dimensions,
      )
      async def _embed(*text: str | Tokens, tensor: np.ndarray | None) -> np.ndarray | None:
        return await self.embed(*text, tensor=tensor)
      setattr(
        self._context.embedding_interface,
        "embed",
        _embed
      )
      self._context.embedding_tokenizer_manager = OpenAITokenizerManager(self.embedding_model.model_id)
      _embeddings_tokenizer_interface = await self._context.embedding_tokenizer_manager.start()

    self._context.concurrent_lock = asyncio.Semaphore(self.concurrency)
    self._context.ratelimit_gate = asyncio.Event()
    self._context.ratelimit_gate.set() # Open the gate
    self._context.is_held = True
    
    logger.info("OpenAI API Session Started")
    return (
      (
        self._context.chat_interface,
        _chat_tokenizer_interface, # type: ignore
      ) if self.chat_model is not None else None,
      (
        self._context.embedding_interface,
        _embeddings_tokenizer_interface, # type: ignore
      ) if self.embedding_model is not None else None,
    )
  
  async def stop(self):
    logger.info("Stopping OpenAI API Session...")
    await self.session.close()
    if self.chat_model is not None:
      assert self._context.chat_tokenizer_manager is not None
      await self._context.chat_tokenizer_manager.stop()
    if self.embedding_model is not None:
      assert self._context.embedding_tokenizer_manager is not None
      await self._context.embedding_tokenizer_manager.stop()
    
    if self._context.ratelimit_task is not None:
      self._context.ratelimit_task.cancel()
      await self._context.ratelimit_task
    
    self._context.is_held = False
    self._context.concurrent_lock = None
    self._context.ratelimit_gate = None
    self._context.chat_interface = None
    self._context.chat_tokenizer_manager = None
    
    logger.info("OpenAI API Session Stopped")
  
  async def rate_limit_requests(self, time: float):
    assert self._context.is_held
    assert self._context.ratelimit_gate is not None

    # Task already running so no need to reschedule
    if self._context.ratelimit_gate.is_set() and self._context.ratelimit_task is not None:
      return

    self._context.ratelimit_task = asyncio.current_task() # Store the task
    try:
      logger.debug(f"Rate Limiting Requests for {time} seconds")
      self._context.ratelimit_gate.clear() # Close the gate
      await asyncio.sleep(time) # Wait for the ratelimit to expire
    except asyncio.CancelledError:
      return # Task was cancelled
    finally:
      self._context.ratelimit_gate.set() # Open the gate
      self._context.ratelimit_task = None

  async def _request(
    self,
    api_path: str,
    data: Any,
    max_attempts: int = 3,
  ) -> dict[str, Any]:
    """Attempt to make a request to the OpenAI API"""
    assert self._context.concurrent_lock is not None
    assert self._context.ratelimit_gate is not None

    logger.debug(f"Making a request to the OpenAI API:\n{json.dumps({'api_path': api_path, 'max_attempts': max_attempts, 'data': data}, indent=2)}")

    total_api_attempts = 0
    while total_api_attempts < max_attempts:
      logger.info(f"API Attempt {total_api_attempts + 1} of 3")
      await self._context.concurrent_lock.acquire()
      await self._context.ratelimit_gate.wait()
      try:
        async with self.session.post(
          api_path,
          json=data,
        ) as chat_resp:
          await _safe_raise_for_status(chat_resp)
          return await chat_resp.json() # type: ignore
      except HTTPError as e:
        logger.error(f"Failed to get a response from the OpenAI API: {e.status} {e.message}\n{e.headers}")
        # Handle Rate Limiting
        if e.status in (429,):
          logger.error(f"Rate Limited by the OpenAI API: {e.headers}")
          rl_wait_time = e.parse_rate_limit_sleep_time()
          logger.info(f"Scheduling a rate limit for {rl_wait_time} seconds...")
          # Schedule a rate limit
          asyncio.create_task(self.rate_limit_requests(rl_wait_time))
          # Sleep to hand control back to the event loop
          await asyncio.sleep(0)
          # Rate limiting doesn't count towards our attempts
        # Raise the Error if it's our fault.
        elif e.status in (400, 401, 403, 404):
          logger.error(f"Failed to get a response from the OpenAI API: {e.status} {e.message}")
          raise Error(e, f"Failed to get a response from the OpenAI API: {e.status} {e.message}")
        # Otherwise Retry
        else:
          logger.warning(f"Got a {e.status} response from the OpenAI API. Will wait a bit & retry...")
          await asyncio.sleep(1)
          total_api_attempts += 1
      finally:
        self._context.concurrent_lock.release()
    
    raise Error(
      None,
      f"Failed to get a response from the OpenAI API after {max_attempts} attempts."
    )

  async def embed(
    self,
    *text: str | Tokens,
    tensor: np.ndarray | None = None,
  ) -> np.ndarray:
    """Embed a batch of text
    Args:
      text (str | Tokens): A batch of text or already encoded Tokens to embed
      tensor (np.ndarray | None): When provided, the embedding vector is stored in this tensor. The tensor must have the shape (len(text), dims) & dtype float32.
    
    Returns:
      The embedding vector of size (len(text), dims) & dtype float32. If tensor is provided, the same tensor is returned.
    """
    assert self._context.is_held
    assert self.embedding_model is not None
    assert self._context.embedding_interface is not None
    assert self._context.embedding_tokenizer_manager is not None
    logger.debug("Embedding...")

    # Convert text into Tokens
    if isinstance(text[0], str):
      assert self._context.chat_tokenizer_manager is not None
      assert all(isinstance(t, str) for t in text)
      text = await self._context.chat_tokenizer_manager.encode(*text) # type: ignore
    else:
      text = list(text) # type: ignore

    if len(text) < 0 or len(text) > self.embedding_model.model_context_window:
      err_msg = f"Text must be between 1 and {self.embedding_model.model_context_window} tokens long"
      logger.error(err_msg)
      raise ModelError(
        IndexError(len(text)),
        err_msg,
      )

    logger.debug(f"Embedding a total of {sum(len(t) for t in text)} tokens")

    # Create the embeddings
    api_response = await self._request(
      "/v1/embeddings",
      {
        "input": text,
      }
    )
    logger.debug(f"Embedding Response:\n{json.dumps(api_response, indent=2)}")

    # Create a Tensor if one wasn't provided
    if tensor is None:
      logger.debug(f"Creating a new Tensor of shape {(len(text), self.embedding_model.vector_dimensions)}")
      tensor = np.empty((len(text), self.embedding_model.vector_dimensions), dtype=np.float32)
    
    # Load the embeddings into the tensor
    embedding_vectors = [d["embedding"] for d in api_response["data"]]
    assert len(embedding_vectors) == len(text)
    assert all(len(v) == self.embedding_model.vector_dimensions for v in embedding_vectors)

    # TODO: Is there a more efficient way of doing this?
    np.copyto(tensor, embedding_vectors)

    return tensor

  async def chat(
    self,
    messages: list[ChatMessage],
    responses: int = 1,
    max_tokens: int | None = None,
  ) -> tuple[ChatMessage, ...]:
    """Submit a Chat, wait for a reponse and return the completed chat
    Provides overrides to the default options for this chat on a per-call basis
    """
    assert self._context.is_held
    assert self.chat_model is not None
    assert self.chat_opts is not None
    assert self._context.chat_interface is not None
    assert self._context.chat_tokenizer_manager is not None
    logger.debug("Chatting...")
    
    # To calculate token count, we need to convert into OpenAI format
    openai_chat_messages: list[dict[str, str]] = [
      {
        "content": chat_message.content,
        "role": chat_message.role,
      } for chat_message in messages
    ]
    logger.debug(openai_chat_messages)
    assert isinstance(openai_chat_messages, list)
    assert all(isinstance(m, dict) for m in openai_chat_messages)
    assert all(isinstance(m.get("content"), str) for m in openai_chat_messages)
    assert all(isinstance(m.get("role"), str) for m in openai_chat_messages)
    assert all(m["role"] in {"system", "user", "assistant", "function"} for m in openai_chat_messages)

    if max_tokens is None:
      # TODO: Calculate a reasonable default to avoid rate limiting
      max_tokens = self.chat_opts.max_tokens
    
    if max_tokens > self.chat_model.model_context_window:
      raise ValueError(f"The configured maximum tokens ({max_tokens}) is greater than the model's context window ({self.chat_model.model_context_window})")

    token_count: int = await self._context.chat_tokenizer_manager.tokens_consumed(*messages)
    remaining_tokens: int = min(
      max_tokens - token_count, # The soft limit
      self.chat_model.model_context_window - token_count, # The hard limit
    )
    _token_count_info = {
      "input_tokens": token_count,
      "tokens_remaining": remaining_tokens,
      "max_configured_output_tokens": max_tokens,
      "model_max_tokens": self.chat_model.model_context_window,
    }
    if remaining_tokens <= 0:
      err_msg = f"No tokens remain for this chat:\n{json.dumps(_token_count_info, indent=2)}"
      logger.error(err_msg)
      raise Error(None, "Chat exceeds maximum token count")
    elif token_count >= self.chat_model.model_context_window // 2:
      logger.warning(f"Chat is approaching maximum token count ({self.chat_model.model_context_window // 2} ≦ {token_count} < {self.chat_model.model_context_window})")

    logger.debug(f"Chat Token Counts:\n{json.dumps(_token_count_info, indent=2)}")

    api_response = await self._request(
      "/v1/chat/completions",
      {
        "model": self.chat_model.model_id,
        "messages": openai_chat_messages,
        "stream": False,
        "n": responses,
        **{ # Override the default options & merge back into the request body
          **self.chat_opts.to_dict(),
          "max_tokens": remaining_tokens,
        },
      },
    )
    unix_ts_ns = time.time_ns()
    chat_responses: list[ChatMessage] = [
      ChatMessage(
        content=reply_obj["message"]["content"],
        role=reply_obj["message"]["role"],
        model=self.chat_model.model_id,
        metadata={ k:v for k, v in {
          "partial": None if reply_obj["finish_reason"] == "stop" else reply_obj["finish_reason"],
          "unix_ts_ns": unix_ts_ns,
        }.items() if v is not None}
      )
      for reply_obj in api_response["choices"]
    ]
    
    return tuple(chat_responses)

  # async def chat(
  #   self,
  #   messages: list[ChatMessage],
  #   responses: int = 1,
  #   max_tokens: int | None = None,
  # ) -> ChatMessage | tuple[ChatMessage, ...]:
  #   """Submit a Chat, wait for a reponse and return the completed chat
  #   Provides overrides to the default options for this chat on a per-call basis
  #   """
  #   assert self._context.is_held
  #   logger.debug("Chatting...")
    
  #   # To calculate token count, we need to convert into OpenAI format
  #   openai_chat_messages: list[dict[str, str]] = [
  #     {
  #       "content": chat_message.content,
  #       "role": chat_message.role,
  #     } for chat_message in messages
  #   ]
  #   logger.debug(openai_chat_messages)
  #   assert self._context.chat_tokenizer_manager is not None
  #   token_count: int = await self._context.chat_tokenizer_manager.tokens_consumed(*messages)
  #   logger.debug(f"Input Tokens Size: {token_count}")
  #   logger.debug(f"Output Tokens Max Size:{(self.chat_opts.max_tokens - token_count )}")
  #   if token_count >= self.chat_opts.max_tokens:
  #     logger.error(f"Chat meets or exceeds the requested token response limit ({token_count} ≧ {self.chat_opts.max_tokens})")
  #     raise Error(None, "Chat exceeds maximum token count")
  #   elif token_count >= self.chat_model.model_context_window // 2:
  #     logger.warning(f"Chat is approaching maximum token count ({self.chat_model.model_context_window // 2} ≦ {token_count} < {self.chat_model.model_context_window})")
  #   elif token_count > self.chat_model.model_context_window:
  #     raise RuntimeError(f"Chat exceeds maximum token count ({token_count} ≧ {self.chat_model.model_context_window})")
    
  #   logger.debug(f"{openai_chat_messages=}")
  #   assert isinstance(openai_chat_messages, list)
  #   assert all(isinstance(m, dict) for m in openai_chat_messages)
  #   assert all(isinstance(m.get("content"), str) for m in openai_chat_messages)
  #   assert all(isinstance(m.get("role"), str) for m in openai_chat_messages)
  #   assert all(m["role"] in {"system", "user", "assistant", "function"} for m in openai_chat_messages)
  #   chat_opts = self.chat_opts.to_dict()
  #   if max_tokens is None:
  #     chat_opts['max_tokens'] = int(min(
  #       chat_opts.get('max_tokens', self.chat_model.model_context_window),
  #       self.chat_model.model_context_window - token_count
  #     ))
  #   else:
  #     chat_opts['max_tokens'] = max_tokens
  #   logger.info(f"Chat Options: {chat_opts}")
  #   req_data = {
  #     "model": self.chat_model.model_id,
  #     "messages": openai_chat_messages,
  #     "stream": False,
  #     "n": responses,
  #     **chat_opts,
  #   }
  #   logger.debug(f"{json.dumps(req_data, indent=2)}")
  #   chat_responses: list[dict] = []
  #   assert self._context.concurrent_lock is not None
  #   assert self._context.ratelimit_gate is not None
  #   total_api_attempts = 0
  #   while total_api_attempts < 3:
  #     logger.info(f"API Attempt {total_api_attempts + 1} of 3")
  #     await self._context.concurrent_lock.acquire()
  #     await self._context.ratelimit_gate.wait()
  #     try:
  #       async with self.session.post(
  #         "/v1/chat/completions",
  #         json=req_data,
  #       ) as chat_resp:
  #         await _safe_raise_for_status(chat_resp)
  #         chat_resp_json = await chat_resp.json()
  #         assert isinstance(chat_resp_json, dict) \
  #           and chat_resp_json.keys() >= {"object", "choices"} \
  #           and chat_resp_json["object"] == "chat.completion" \
  #           , "Invalid response from OpenAI API"
  #         assert len(chat_resp_json["choices"]) > 0, "Invalid response from OpenAI API: Got no choices"
  #         chat_responses = chat_resp_json["choices"]
  #         logger.info(f"Final Token Consumption: {chat_resp_json['usage']['total_tokens']} Tokens ({chat_resp_json['usage']['prompt_tokens']} prompt + {chat_resp_json['usage']['completion_tokens']} completion)")
  #         break
  #     except HTTPError as e:
  #       logger.error(f"Failed to get a response from the OpenAI API: {e.status} {e.message}\n{e.headers}")
  #       # Handle Rate Limiting
  #       if e.status in (429,):
  #         logger.error(f"Rate Limited by the OpenAI API: {e.headers}")
  #         rl_wait_time = e.parse_rate_limit_sleep_time()
  #         logger.info(f"Scheduling a rate limit for {rl_wait_time} seconds...")
  #         # Schedule a rate limit
  #         asyncio.create_task(self.rate_limit_requests(rl_wait_time))
  #         # Rate limiting doesn't count towards our attempts
  #       # Raise the Error if it's our fault.
  #       elif e.status in (400, 401, 403, 404):
  #         logger.error(f"Failed to get a response from the OpenAI API: {e.status} {e.message}")
  #         raise Error(e, f"Failed to get a response from the OpenAI API: {e.status} {e.message}")
  #       # Otherwise Retry
  #       else:
  #         logger.warning(f"Got a {e.status} response from the OpenAI API. Retrying...")
  #         logger.info(f"Sleeping for {1} seconds...")
  #         await asyncio.sleep(1)
  #         total_api_attempts += 1
  #     finally:
  #       self._context.concurrent_lock.release()
    
  #   if len(chat_responses) == 0:
  #     raise Error(None, "Failed to get a response from the OpenAI API 🤷‍♂️")
  #   assert all(isinstance(resp, dict) for resp in chat_responses) \
  #     and all(resp.keys() >= {"message", "finish_reason"} for resp in chat_responses)

  #   if any(resp["finish_reason"] != "stop" for resp in chat_responses):
  #     logger.warning(f"Chat did not finish normally b/c reason '{[resp['finish_reason'] for resp in chat_responses if resp['finish_reason'] != 'stop']}'")
        
  #   # assert isinstance(chat_response["message"], dict) \
  #   #   and chat_response["message"].keys() >= {"content", "role"}

  #   assert all(isinstance(resp["message"], dict) for resp in chat_responses) \
  #     and all(resp["message"].keys() >= {"content", "role"} for resp in chat_responses)
    
  #   if len(chat_responses) == 1:
  #     return ChatMessage(
  #       content=chat_responses[0]["message"]["content"],
  #       role=chat_responses[0]["message"]["role"],
  #       model=self.chat_model.model_id,
  #     )

  #   return tuple(
  #     ChatMessage(
  #       content=resp["message"]["content"],
  #       role=resp["message"]["role"],
  #       model=self.chat_model.model_id,
  #     )
  #     for resp in chat_responses
  #   )
