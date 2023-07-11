"""Render the content of Chat Markup Language (CML) to raw text.
"""
import os
import sys
import asyncio
from collections import ChainMap
from typing import Any
from loguru import logger
import json

from src import lexer as lex

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

  if kwargs["rich"]:
    sys.stdout.write("---\n\n")
  for node in chat_ast:
    if isinstance(node, lex.MessageNode):
      logger.trace(f"Node Metadata Props: {node.metadata.props}")
      if kwargs["rich"]:
        sys.stdout.write(f"> {node.metadata.props['role'].upper()}\n\n")
      sys.stdout.write(node.content.content.decode(node.content.encoding))
      if kwargs["rich"]:
        sys.stdout.write("\n\n---")
      sys.stdout.write("\n\n")

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
    "rich": False,
  }
  for arg in args:
    if arg.startswith("-"):
      try:
        key, value = arg.split("=")
      except ValueError:
        key = arg
        value = True
      key = key.lstrip('-').lower()
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

Print the metadata associated with each message in a chat.

Options:
  -h, --help                         Show this help message and exit.  
  --verbose                          Enable verbose logging.
  --debug                            Enable debug logging.
  --trace                            Enable trace logging.
  --quiet                            Disable all logging.
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
