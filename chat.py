import asyncio
import json
import os
import pathlib
import re
import sys
from collections import ChainMap, namedtuple
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List

import aiohttp
import tiktoken
from loguru import logger

import lexer as lex

AVAILABLE_MODELS = {
  "gpt4": {
    "id": "gpt-4-0613",
    "max_tokens": 8192,
  },
  "gpt3": {
    "id": "gpt-3.5-turbo-0613",
    "max_tokens": 4096,
  },
  "gpt3-16k": {
    "id": "gpt-3.5-turbo-16k-0613",
    "max_tokens": 16384,
  }
}
MODEL_TUNING_PRESETS = {
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
class ChatOptions:
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
class Model:
  id: str
  max_tokens: int

@dataclass
class ChatSessionManager:
  model: Model
  reverse: bool
  opts: ChatOptions
  session: aiohttp.ClientSession = field(default_factory=aiohttp.ClientSession)
  lexer: lex.ChatLexer = field(default_factory=lex.ChatLexer)
  parser: lex.TokenParser = field(default_factory=lex.TokenParser)

  # _context: _ChatContext = field(init=False, default_factory=_ChatContext)

  async def __aenter__(self):
    await self.start()
  
  async def __aexit__(self, exc_type, exc_value, traceback):
    await self.stop()

  async def start(self):
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
        logger.trace(models_resp_json)
        model_list = [
          m['id'] for m in models_resp_json["data"]
          if m["object"] == "model"
        ]
        
      assert len(model_list) > 0, "No models found"
      if self.model.id not in model_list:
        logger.error("Couldn't find the requested model")
        raise ValueError(f"Model {self.model.id} not found. Available models: {model_list}")
    except Error as e:
      logger.error(f"Failed to start Chat Session: {e}")
      await self.session.close()
      raise
    logger.success("Chat Session Started")
  
  async def stop(self):
    logger.info("Stopping Chat Session...")
    logger.debug("Closing OpenAI API Session...")
    await self.session.close()
    logger.success("Chat Session Stopped")
  
  async def chat(self, chat: str) -> str:
    """Submit a Chat, wait for a reponse and return the completed chat"""
    logger.debug("Chatting...")
    
    chat_ast = list(self.parser.parse(self.lexer.tokenize(chat)))
    chat_messages = lex.transpile_ast_to_message_dict(*chat_ast)
    assert all(isinstance(m, dict) for m in chat_messages), "Invalid chat message parsed"
    assert all(isinstance(m["content"], str) for m in chat_messages), "Invalid chat message parsed"
    assert all(isinstance(m["metadata"], dict) for m in chat_messages), "Invalid chat message parsed"
    assert all({'role'} <= m["metadata"].keys() for m in chat_messages), "Invalid chat message parsed"
    logger.trace(chat_messages)
    # To calculate token count, we need to convert into OpenAI format
    openai_chat_messages: list[dict[str, str]] = [
      {
        "content": chat_message['content'],
        "role": chat_message['metadata']['role'] # type: ignore
      } for chat_message in chat_messages
    ]
    logger.trace(openai_chat_messages)
    token_count_by_message = calculate_tokens(self.model.id, openai_chat_messages)
    token_count = sum(token_count_by_message)
    logger.debug(f"Total Token Count: {token_count}")
    logger.trace(f"{(token_count + self.opts.max_tokens)=}")
    if token_count >= self.opts.max_tokens:
      logger.error(f"Chat meets or exceeds the requested token response limit ({token_count} ≧ {self.opts.max_tokens})")
      raise Error(None, "Chat exceeds maximum token count")
    elif token_count >= self.model.max_tokens // 2:
      logger.warning(f"Chat is approaching maximum token count ({self.model.max_tokens // 2} ≦ {token_count} < {self.model.max_tokens})")
    elif token_count > self.model.max_tokens:
      raise RuntimeError(f"Chat exceeds maximum token count ({token_count} ≧ {self.model.max_tokens})")
    
    # Reverse Roles if requested
    if self.reverse:
      openai_chat_messages = list(map(
        lambda m: {
          "content": m['content'],
          "role": {
            'user': 'assistant',
            'assistant': 'user',
            'system': 'system'
          }[m['role']]
        },
        openai_chat_messages
      ))
    logger.trace(f"{openai_chat_messages=}")
    chat_opts = self.opts.to_dict()
    chat_opts['max_tokens'] = int(min(
      chat_opts.get('max_tokens', self.model.max_tokens),
      self.model.max_tokens - token_count
    ))
    logger.trace(f"{(chat_opts['max_tokens'] + token_count)=}")
    logger.info(f"Chat Options: {chat_opts}")
    req_data = {
      "model": self.model.id,
      "messages": openai_chat_messages,
      "stream": False,
      **chat_opts,
    }
    logger.trace(f"{req_data=}")
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
    
    if self.reverse:
      chat_response["message"]["role"] = {
        'user': 'assistant',
        'assistant': 'user',
        'system': 'system'
      }[chat_response["message"]["role"]]
    
    assert isinstance(chat_response["message"], dict) \
      and chat_response["message"].keys() >= {"content", "role"}

    # Add the response to the chat Messages
    chat_messages.append({
      "content": chat_response["message"]["content"],
      "metadata": {
        "role": chat_response["message"]["role"],
        "model": self.model.id,
      },
    })
    # Reverse transpile the chat messages back into the chat AST
    chat_ast = lex.transpile_message_dict_to_ast(*chat_messages)

    # Render the Markdown
    return lex.transpile_ast_to_markdown(*chat_ast)

async def _main(*args: str, **kwargs: Any) -> int:
  logger.trace(f"Args: {args}")
  logger.trace(f"Kwargs: {kwargs}")
  
  openai_api_session = aiohttp.ClientSession(
    base_url="https://api.openai.com",
    headers={k: v for k, v in {
      "Authorization": f"Bearer {kwargs['openai_api_key']}",
      "OpenAI-Organization": kwargs.get("openai_org_id", None),
      # "Content-Type": "application/json",
    }.items() if v is not None},
  )

  chat_model = Model(**AVAILABLE_MODELS[kwargs["model"]])  

  # Chat Options are merged with the following precedence:
  # 1. CLI Overrides
  # 2. Personality Presets
  # 3. (Implicit) Defaults
  chat_opts = ChatOptions(**ChainMap(*[
    opts for opts in [
      # CLI Overrides
      {k: v for k, v in {
        "temperature": kwargs.get("temperature", None),
        "max_tokens": kwargs.get("tokens", None),
        "top_p": kwargs.get("topp", None),
      }.items() if v is not None},
      # Personality Defaults
      MODEL_TUNING_PRESETS[kwargs["personality"]]["tuning"] if "personality" in kwargs else None,
      # Defaults,
      {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": chat_model.max_tokens,
      }
    ] if opts is not None
  ]))

  chat_session_manager = ChatSessionManager(
    model=chat_model,
    reverse=kwargs["reverse"],
    opts=chat_opts,
    session=openai_api_session,
  )

  chat = sys.stdin.read().strip()
  if not chat:
    raise RuntimeError("Chat must not be empty")
  
  async with chat_session_manager:
    response = await chat_session_manager.chat(chat)
  sys.stdout.write(response)

  return 0

def _parse_env(**env: str) -> dict[str, Any]:
  return {
    kwarg_key: env[env_key]
    for env_key, kwarg_key in {
      "OPENAI_API_KEY": "openai_api_key",
      "OPENAI_ORG_ID": "openai_org_id",
    }.items()
    if env_key in env and env[env_key] is not None and env[env_key] != ""
  }

def _parse_kwargs(*args: str) -> dict[str, Any]:
  _kwargs: dict[str, Any] = {
    "help": False,
    "verbose": False,
    "debug": False,
    "trace": False,
    "quiet": False,
    "model": "gpt3",
    "personality": None,
    "temperature": None,
    "tokens": None,
    "topp": None,
    "reverse": False,
  }
  for arg in args:
    if arg.startswith("-"):
      try:
        key, value = arg.split("=")
      except ValueError:
        key = arg
        value = True
      key = key.lstrip('-').lower()
      if isinstance(value, str):
        if key == "tokens":
          value = int(value)
        elif key in {"temperature", "topp"}:
          value = float(value)
      _kwargs[key] = value
  return {k: v for k, v in _kwargs.items() if v is not None}

def _parse_args(*args: str) -> list:
  _args = []
  for arg in args:
    if not arg.startswith("-"):
      _args.append(arg)
  return _args

def _help():
  print(f"""
Usage: {sys.argv[0]} [OPTIONS]

This script provides a chatbot interface using the OpenAI API. You can select different models and modify chat options for customized responses. The chatbot will read from stdin and write to stdout. Useful for chaining prompts.

Options:
  -h, --help                         Show this help message and exit.

  --model=MODEL_ID                   Select the AI model. Available models: gp4, gpt3.5
                                     (Default: gpt3.5)

  --personality=PERSONALITY_ID       Select a personality preset for AI chat responses.
                                     Available presets: creative, balanced, reserved.
                                     (Default: balanced)

  --temperature=FLOAT_VALUE          Set a custom temperature value for the AI chat responses.
                                     (Overwrites the personality preset temperature value.)

  --tokens=INT_VALUE                 Set a custom maximum token count for the AI chat responses.
                                     (Overwrites the personality preset token value.)

  --topp=FLOAT_VALUE                 Set a custom top_p value for the AI chat responses.
                                     (Overwrites the personality preset top_p value.)
  
  --reverse                          Reverse the Roles of the AI and the User in the Chat.

  --verbose                          Enable verbose logging.
  --debug                            Enable debug logging.
  --trace                            Enable trace logging.
  --quiet                            Disable all logging.

Environment Variables:
  OPENAI_API_KEY                     Set the OpenAI API key for authentication.
  OPENAI_ORGANIZATION                Set the OpenAI organization for API access.

Examples:

  1. Using the script with default options and personality preset:
     cat ./chat.md | python {sys.argv[0]} --model=gpt3 --personality=balanced > ./completed_chat.md

  2. Customizing chat options for a more creative response:
     cat ./chat.md | python {sys.argv[0]} --model=gpt3 --temperature=1.5 --topp=0.9 --tokens=4096 > ./completed_chat.md
  """) 

def _setup_logging(**_kwargs):
  script_log_level = 'WARNING'
  if _kwargs["verbose"]:
    script_log_level="INFO"
  elif _kwargs["debug"]:
    script_log_level="DEBUG"
  elif _kwargs["trace"]:
    script_log_level="TRACE"
  
  if not _kwargs["quiet"]:  
    logger.add(sys.stderr, level=script_log_level, enqueue=True)

if __name__ == "__main__":
  _rc = 255
  try:
    logger.remove()
    # logger.add(sys.stderr, level='TRACE')
    _args = _parse_args(*sys.argv[1:])
    _kwargs = ChainMap({}, # ChainMaps are shallow copies, so make sure to capture any writes to prevent overwriting.
      _parse_kwargs(*sys.argv[1:]),
      _parse_env(**os.environ),
    )
    _setup_logging(**_kwargs)
    if _kwargs["help"]:
      _help()
      _rc = 0
    else:
      _rc = asyncio.run(_main(*_args, **_kwargs))
  except Exception as e:
    logger.opt(exception=e).critical("Unhandled Exception raised during runtime...")
  finally:
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
