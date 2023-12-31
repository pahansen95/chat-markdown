"""The Chat CLI

Chat with an LLM from the command line using Markdown Files.
"""
import os
import sys
import asyncio
from collections import ChainMap
from typing import Any
import aiohttp
from loguru import logger

from src import openai, lexer as lex, response as resp
from src.interfaces import ChatMessage

async def _main(*args: str, **kwargs: Any) -> int:
  logger.trace(f"Args: {args}")
  logger.trace(f"Kwargs: {kwargs}")

  # Parse the Input
  chat = sys.stdin.read().strip()
  if not chat:
    raise RuntimeError("Chat must not be empty")
  
  lexer: lex.ChatLexer = lex.ChatLexer()
  parser: lex.TokenParser = lex.TokenParser()

  chat_ast = list(parser.parse(lexer.tokenize(chat)))
  chat_message_objs = lex.transpile_ast_to_message_dict(*chat_ast)
  assert all(isinstance(m, dict) for m in chat_message_objs), "Invalid chat message parsed"
  assert all(isinstance(m["content"], str) for m in chat_message_objs), "Invalid chat message parsed"
  assert all(isinstance(m["metadata"], dict) for m in chat_message_objs), "Invalid chat message parsed"
  assert all({'role'} <= m["metadata"].keys() for m in chat_message_objs), "Invalid chat message parsed"
  logger.debug(f"{chat_message_objs=}")

  # TODO: Cut out the middleman & use the AST directly
  chat_messages = [
    ChatMessage(
      content=message["content"], # type: ignore
      role=message["metadata"]["role"], # type: ignore
      model=message["metadata"].get("model", None), # type: ignore
      metadata={
        k: v
        for k, v in message["metadata"].items() # type: ignore
        if k not in {"role", "model"}
      },
    )
    for message in chat_message_objs
  ]

  # Setup session with the LLM
  
  openai_api_session = aiohttp.ClientSession(
    base_url="https://api.openai.com",
    headers={k: v for k, v in {
      "Authorization": f"Bearer {kwargs['openai_api_key']}",
      "OpenAI-Organization": kwargs.get("openai_org_id", None),
      "Content-Type": "application/json",
    }.items() if v is not None},
  )
  _model = openai.OPENAI_AVAILABLE_MODELS[kwargs["model"]]
  openai_model_info = openai.OpenAIModelInfo(
    model_id=_model["id"],
    model_context_window=_model["context_window"],
  )

  # Chat Options are merged with the following precedence:
  # 1. CLI Overrides
  # 2. Personality Presets
  # 3. (Implicit) Defaults
  openai_chat_opts = openai.OpenAIChatOptions(**ChainMap(*[
    opts for opts in [
      # CLI Overrides
      {k: v for k, v in {
        "temperature": kwargs.get("temperature", None),
        "max_tokens": kwargs.get("tokens", None),
        "top_p": kwargs.get("topp", None),
      }.items() if v is not None},
      # Personality Defaults
      openai.OPENAI_MODEL_TUNING_PRESETS[kwargs["personality"]]["tuning"] if "personality" in kwargs else None,
      # Defaults,
      {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": openai_model_info.model_context_window,
      }
    ] if opts is not None
  ]))

  chat_session_manager = openai.OpenAISessionManager(
    chat_model=openai_model_info,
    chat_opts=openai_chat_opts,
    session=openai_api_session,
  )
  total_responses = max(int(kwargs["replies"]), 1)
  model_responses: tuple[ChatMessage, ...] = ()
  async with chat_session_manager as model_tools:
    assert model_tools[0] is not None
    if kwargs["mode"] == "ss":
      logger.debug("Single Shot Mode")
      model_responses = await resp.single_shot(
        messages=chat_messages,
        llm=model_tools[0][0],
        responses=total_responses,
      )
    elif kwargs["mode"] == "cot":
      logger.debug("Chain of Thought Mode")
      if total_responses > 1:
        logger.warning(f"Chain of Thought Mode does not support multiple responses. '--responses={total_responses}' will be ignored.")
        total_responses = 1
      # TODO: Allow overriding the CoT LLM (e.g. for a specialized LLM or one w/ a larger context window?)
      # But for now just create a second LLM using gpt3.5 to use for the CoT to keep things cheaper
      cot_openai_api_session = aiohttp.ClientSession(
        base_url="https://api.openai.com",
        headers={k: v for k, v in {
          "Authorization": f"Bearer {kwargs['openai_api_key']}",
          "OpenAI-Organization": kwargs.get("openai_org_id", None),
          # "Content-Type": "application/json",
        }.items() if v is not None},
      )
      # TODO: While GPT3.5 is cheaper than GPT3.5-16k it still doesn't pay very strong attention to the system message which negatively impacts response quality
      cot_model_info = openai.OpenAIModelInfo(
        model_id=openai.OPENAI_AVAILABLE_MODELS["gpt3-16k"]["id"],
        model_context_window=openai.OPENAI_AVAILABLE_MODELS["gpt3-16k"]["context_window"],
      )
      cot_session_manager = openai.OpenAISessionManager(
        chat_model=cot_model_info,
        chat_opts=openai_chat_opts,
        session=cot_openai_api_session,
      )
      async with cot_session_manager as cot_model_tools:
        assert cot_model_tools[0] is not None
        model_responses = await resp.chain_of_thought(
          messages=chat_messages,
          llm=model_tools[0][0],
          cot_llm=cot_model_tools[0][0],
        )
    elif kwargs["mode"] == "tot":
      raise NotImplementedError("Tree of Thoughts is not yet implemented")
    else:
      raise RuntimeError(f"Invalid mode: {kwargs['mode']}")

  assert len(model_responses) > 0, "No responses generated"
  assert total_responses == len(model_responses), f"Requested {total_responses} responses but only received {len(model_responses)}"
  # TODO: Cut out the middleman & use the AST directly
  response = lex.transpile_ast_to_markdown(*[
    *chat_ast,
    *lex.transpile_message_dict_to_ast(*[{
      "content": resp.content,
      "metadata": {
        "role": resp.role,
        "model": resp.model, # type: ignore
        "mode": {
          "ss": "Single Shot",
          "cot": "Chain of Thought",
          "tot": "Tree of Thoughts",
        }[kwargs["mode"]],
        **resp.metadata,
      },
    } for resp in model_responses]),
  ])
  
  # Add the user role to the response for convenience
  sys.stdout.write(f"""{response.strip()}

---

<!-- start {{"role": "user"}} -->

<!-- end -->
""")

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
    "model": "gpt3-16k",
    "personality": None,
    "temperature": None,
    "tokens": None,
    "topp": None,
    "reverse": False,
    "mode": "ss",
    "replies": 1,
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

  --model=MODEL_ID                   Select the AI model. Available models: gp4, gpt3-16k, gpt3
                                     (Default: gpt3-16k)

  --mode=RESPONSE_MODE               Select how the model generates a response.
                                     Available modes: ss, cot, tot
                                     (Default: ss)
  
  --personality=PERSONALITY_ID       Select a personality preset for AI chat responses.
                                     Available presets: creative, balanced, reserved.
                                     (Default: balanced)

  --temperature=FLOAT_VALUE          Set a custom temperature value for the AI chat responses.
                                     (Overwrites the personality preset temperature value.)

  --tokens=INT_VALUE                 Set a custom maximum token count for the AI chat responses.
                                     (Overwrites the personality preset token value.)

  --topp=FLOAT_VALUE                 Set a custom top_p value for the AI chat responses.
                                     (Overwrites the personality preset top_p value.)
  
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

Response Modes:
  ss - Single Shot
    This is the standard response mode & generally the fastest. The LLM will generate a single, direct response to the input.
    Use Single Shot when you want a quick response, have a simple prompt or are exploring a topic or thought process.

  cot - Chain of Thought
    Take more time & generate a thoughtful response. CoT is a framework that enhances an LLM's problem-solving capabilities by employing an ordered sequence of reasoning steps that collectively lead to a (more) comprehensive solution. For more information see https://arxiv.org/pdf/2201.11903.pdf
    Use Chain of Thought when you want a more comprehensive response, have a complex prompt or have a problem statement that is well defined, well scoped & has concrete actionables/questions. CoT is best applied after narrowing down the scope using Single Shot.

  tot - Tree of Thoughts
    NOT YET IMPLEMENTED
    Take alot of time & search the entire solution space for a more optimal response. [ADD MORE DESCRIPTION HERE]
    Use Tree of Thoughts when [TBD]
  """) 

def _setup_logging(**_kwargs):
  script_log_level = 'WARNING'
  if _kwargs["verbose"]:
    script_log_level="INFO"
  elif _kwargs["debug"]:
    script_log_level="DEBUG"
  elif _kwargs["trace"]:
    script_log_level="TRACE"
    # Setup Python Logging to print to stderr
    import logging
    logging.basicConfig(level=logging.DEBUG)
  
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
