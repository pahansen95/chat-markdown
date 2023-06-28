"""# LLM Module

Implements an interface for prompting LLMs.

Right now we are tightly coupled to OpenAI's LLMs & API but we expect to support locally running OpenSource models via GGML in the near future.
"""

import asyncio
import aiohttp
import logging
import tiktoken

from dataclasses import dataclass, field
from typing import Any

from src._interfaces import ChatMessage

from ._interfaces import ChatMessage, LLM

logger = logging.getLogger(__name__)

OPENAI_AVAILABLE_MODELS = {
  "gpt4": {
    "id": "gpt-4-0613",
    "context_window": 8192,
  },
  "gpt3": {
    "id": "gpt-3.5-turbo-0613",
    "context_window": 4096,
  },
  "gpt3-16k": {
    "id": "gpt-3.5-turbo-16k-0613",
    "context_window": 16384,
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

class Error(Exception):
  def __init__(self, exc: Exception | None, message: str):
    super().__init__(message)
    self.exc = exc

class HTTPError(Error):
  def __init__(self, exc: Exception | None, status: int, message: str):
    super().__init__(exc=exc, message=message)
    self.status = status
    self.message = message
  
async def _safe_raise_for_status(response: aiohttp.ClientResponse) -> None:
  try:
    response.raise_for_status()
  except aiohttp.ClientResponseError as cre:
    # Format the response JSON encodable dict
    _data = {
      "code": cre.status,
      "message": cre.message,
      "headers": dict(response.headers),
    }
    logger.info(f"Response: {_data}")
    logger.error(f"HTTP Error: {cre.status} - {cre.message}")
    raise HTTPError(None, cre.status, cre.message)

# def _calc_tokens(model: str, message: str) -> int:
#   try:
#     encoding = tiktoken.encoding_for_model(model)
#   except KeyError:
#     logger.warning(f"Model {model} not found in TikToken. Using default encoding.")
#     encoding = tiktoken.get_encoding("cl100k_base")
  
#   return len(encoding.encode(message))

# see https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
def _calc_tokens(model: str, *messages: dict[str, str]):
    """Returns the number of tokens used by a list of messages."""
    # TODO: I need to re-implement from scratch to understand what's happening behind the scenes. I'm pretty sure it has to do w/ OpenAI's ChatML format: https://github.com/openai/openai-python/blob/main/chatml.md
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    if "gpt-3.5-turbo" in model:
      tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
      tokens_per_name = -1  # if there's a name, the role is omitted
      offset = 0
    elif "gpt-4" in model:
      tokens_per_message = 3
      tokens_per_name = 1
      offset = 1
    else:
      raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
      num_tokens += tokens_per_message
      for key, value in message.items():
        num_tokens += len(encoding.encode(value))
        if key == "name":
          num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    num_tokens += offset
    return num_tokens

def calculate_tokens(model: str, messages: list[dict[str, str]]) -> list[int]:
  return [
    _calc_tokens(
      model,
      chat_message
    )
    for chat_message in messages
  ]

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
class _OpenAISessionManagerContext:
  is_held: bool = False
  llm_interface: LLM | None = None

@dataclass
class OpenAISessionManager:
  model: OpenAIModelInfo
  opts: OpenAIChatOptions
  session: aiohttp.ClientSession = field(default_factory=aiohttp.ClientSession)
  _context: _OpenAISessionManagerContext = field(init=False, default_factory=_OpenAISessionManagerContext)
  
  async def __aenter__(self) -> LLM:
    return await self.start()
  
  async def __aexit__(self, exc_type, exc_value, traceback):
    await self.stop()

  async def start(self) -> LLM:
    logger.info("Starting Chat Session...")
    # Test Authorization
    logger.debug("Testing OpenAI API Authorization...")
    try:
      async with self.session.get("/v1/models") as models_resp:
        await _safe_raise_for_status(models_resp)
        models_resp_json = await models_resp.json()
        assert isinstance(models_resp_json, dict) \
          and models_resp_json.keys() >= {"object", "data"} \
          and models_resp_json["object"] == "list" \
          , "Invalid response from OpenAI API"
        logger.debug(models_resp_json)
        model_list = [
          m['id'] for m in models_resp_json["data"]
          if m["object"] == "model"
        ]
        
      assert len(model_list) > 0, "No models found"
      if self.model.model_id not in model_list:
        logger.error("Couldn't find the requested model")
        raise ValueError(f"Model {self.model.model_id} not found. Available models: {model_list}")
    except Error as e:
      logger.error(f"Failed to start Chat Session: {e}")
      await self.session.close()
      raise
    self._context.is_held = True
    llm_interface = LLM(
      name=self.model.model_id,
      context_window=self.model.model_context_window,
    )
    async def _calc_tokens(*messages: ChatMessage) -> list[int]:
      return calculate_tokens(
        model=self.model.model_id,
        messages=[
          {
            "content": chat_message.content,
            "role": chat_message.role,
          } for chat_message in messages
        ]
      )
    setattr(
      llm_interface,
      "calculate_tokens",
      _calc_tokens
    )
    async def _chat(messages: list[ChatMessage]) -> ChatMessage:
      return await self.chat(messages)
    setattr(
      llm_interface,
      "chat",
      _chat
    )
    self._context.llm_interface = llm_interface
    logger.info("Chat Session Started")
    assert self._context.llm_interface is not None
    return self._context.llm_interface
  
  async def stop(self):
    logger.info("Stopping Chat Session...")
    logger.debug("Closing OpenAI API Session...")
    await self.session.close()
    self._context.is_held = False
    self._context.llm_interface = None
    logger.info("Chat Session Stopped")
  
  async def chat(self, messages: list[ChatMessage]) -> ChatMessage:
    """Submit a Chat, wait for a reponse and return the completed chat"""
    assert self._context.is_held
    logger.debug("Chatting...")
    
    # To calculate token count, we need to convert into OpenAI format
    openai_chat_messages: list[dict[str, str]] = [
      {
        "content": chat_message.content,
        "role": chat_message.role,
      } for chat_message in messages
    ]
    logger.debug(openai_chat_messages)
    token_count_by_message = calculate_tokens(self.model.model_id, openai_chat_messages)
    token_count = sum(token_count_by_message)
    logger.debug(f"Total Token Count: {token_count}")
    logger.debug(f"{(token_count + self.opts.max_tokens)=}")
    if token_count >= self.opts.max_tokens:
      logger.error(f"Chat meets or exceeds the requested token response limit ({token_count} ≧ {self.opts.max_tokens})")
      raise Error(None, "Chat exceeds maximum token count")
    elif token_count >= self.model.model_context_window // 2:
      logger.warning(f"Chat is approaching maximum token count ({self.model.model_context_window // 2} ≦ {token_count} < {self.model.model_context_window})")
    elif token_count > self.model.model_context_window:
      raise RuntimeError(f"Chat exceeds maximum token count ({token_count} ≧ {self.model.model_context_window})")
    
    logger.debug(f"{openai_chat_messages=}")
    chat_opts = self.opts.to_dict()
    chat_opts['max_tokens'] = int(min(
      chat_opts.get('max_tokens', self.model.model_context_window),
      self.model.model_context_window - token_count
    ))
    logger.debug(f"{(chat_opts['max_tokens'] + token_count)=}")
    logger.info(f"Chat Options: {chat_opts}")
    req_data = {
      "model": self.model.model_id,
      "messages": openai_chat_messages,
      "stream": False,
      **chat_opts,
    }
    logger.debug(f"{req_data=}")
    chat_response = None
    for _ in range(3):
      try:
        async with self.session.post(
          "/v1/chat/completions",
          json=req_data,
        ) as chat_resp:
          await _safe_raise_for_status(chat_resp)
          chat_resp_json = await chat_resp.json()
          assert isinstance(chat_resp_json, dict) \
            and chat_resp_json.keys() >= {"object", "choices"} \
            and chat_resp_json["object"] == "chat.completion" \
            , "Invalid response from OpenAI API"
          assert len(chat_resp_json["choices"]) == 1, "Invalid response from OpenAI API: Got more than one choice"
          chat_response = chat_resp_json["choices"][0]
      except HTTPError as e:
        # Raise the Error if it's our fault.
        if e.status in (400, 401, 403, 404, 429):
          raise Error(e, f"Failed to get a response from the OpenAI API: {e.status} {e.message}")
        logger.warning(f"Got a {e.status} response from the OpenAI API. Retrying...")
        await asyncio.sleep(1)
    
    if chat_response is None:
      raise Error(None, "Failed to get a response from the OpenAI API 🤷‍♂️")
    assert isinstance(chat_response, dict) \
      and chat_response.keys() >= {"message", "finish_reason"}

    if chat_response["finish_reason"] != "stop":
      logger.warning(f"Chat did not finish normally b/c reason '{chat_response['finish_reason']}'")
        
    assert isinstance(chat_response["message"], dict) \
      and chat_response["message"].keys() >= {"content", "role"}

    return ChatMessage(
      content=chat_response["message"]["content"],
      role=chat_response["message"]["role"],
      model=self.model.model_id,
    )
