"""# Lexer Module

Implements a lexical analyzer for the Chat Language.

The Chat Language is just Markdown structured in a specific way to encourage:

- Human Readability & Writeability
- Machine Readability & Writeability
- Low Effort Knowledge Base Creation

By breaking up chats into structured markdown files & embedding metadata in HTML comments, we can quickly build knowledge bases while still providing a human friendly interface for chatting with the LLM.

The basic structure of a chat is as follows:

```markdown
<!-- start {"role": "system"} -->
A System Message
<!-- end -->

---

<!-- start {"role": "user"} -->
A User Message
<!-- end -->

---

<!-- start {"role": "assistant"} -->
An Assistant Message
<!-- end -->
```

The first message must be a system message. The System message may only appear once.
Subsequent messages can be either a user message or an assistant message. Ordering and frequency of user and assistant messages is not restricted.
Messages are wrapped in `start` & `end` blocks which are HTML comments. The `start` block contains metadata in JSON format.
Messages are seperated by a horizontal rule (`---`).

Currently White Space is important (outside of the message content) and must be preserved. This will probably change in the future.

As of now the only limitation on the chat's length is the context window of the LLM.

"""
import re
import json
from typing import Generator, Any, Generic, TypeVar
from collections import namedtuple
from dataclasses import dataclass, field
from abc import ABC
import logging
import enum

logger = logging.getLogger(__name__)

### The AST ###

class Node(ABC):
  pass

@dataclass
class MetadataNode(Node):
  props: dict[str, Any]

@dataclass
class ContentNode(Node):
  content: bytes
  encoding: str
  pos: tuple[int, int] # line, column

@dataclass
class MessageNode(Node):
  metadata: MetadataNode
  content: ContentNode

### The Lexical Analyzer ###

class TOKEN_KIND(enum.IntEnum):
  START_COMMENT = 0
  END_COMMENT = enum.auto()
  HORIZONTAL_RULE = enum.auto()
  WHITESPACE = enum.auto()
  NEWLINE = enum.auto()
  TEXT = enum.auto()
  EOF = enum.auto()

class TEXT_ENCODING(enum.Enum):
  UTF8 = 'utf-8'

@dataclass
class Token:
  kind: TOKEN_KIND
  value: bytes
  pos: tuple[int, int] # line, column
  encoding: TEXT_ENCODING

  def __str__(self) -> str:
    if self.encoding == TEXT_ENCODING.UTF8:
      # return f"Token(kind='{self.kind.name}', pos='{self.pos}', value='{''.join([f'U+{ord(c):04X}' for c in self.value])}')"
      return f"Token(kind='{self.kind.name}', pos='{self.pos}', value='{self.value.decode('utf-8').encode('unicode_escape').decode('utf-8')}')"
    raise NotImplementedError(f"Encoding '{self.encoding}' is not supported")

  def __repr__(self) -> str:
    return str(self)

@dataclass
class ChatLexer:
  """Lexical Analyzer for the Chat Log Language.
  The Chat Log Lanaguage consists of the following rules:
    - Messages are seperated by a horizontal rule (`\\n---\\n`)
    - Messages are surrounded by the HTML comments (`<!-- start -->` and `<!-- end -->`)
    - The start comment contains metadata in JSON format
      - example: `<!-- start {"role": "assistant", "model": "gpt-4"} -->`
    - The end comment is empty
    - The content is the text between the start and end comments
  """
  patterns: list[tuple[TOKEN_KIND, str]] = field(default_factory=lambda : list([
    ### Order matters here ###
    (TOKEN_KIND.START_COMMENT, r'<!-- start ((?!-->).)* -->'),  # Start comment with metadata
    (TOKEN_KIND.END_COMMENT, r'<!-- end -->'),  # End comment
    (TOKEN_KIND.HORIZONTAL_RULE, r'\n---\n'),  # Message separator
    # Check for Newlines first before Whitespace
    (TOKEN_KIND.NEWLINE, r'\n'),  # One newline
    (TOKEN_KIND.WHITESPACE, r'\s+'),  # One or more spaces
    (TOKEN_KIND.TEXT, r'.+'),  # Any text
  ]))
  
  def tokenize(
    self,
    text: str,
  ) -> Generator[Token, None, None]:
    index = 0
    line = 1
    column = 1
    regex: dict[TOKEN_KIND, re.Pattern] = {
      token_kind: re.compile(pattern, re.MULTILINE)
      for token_kind, pattern in self.patterns
    }
    while index < len(text):
      match = None
      for token_kind, _ in self.patterns:
        match = regex[token_kind].match(text, index, len(text))
        if match is not None:
          value = match.group(0)
          token = Token(
            kind=token_kind,
            value=value.encode('utf-8'),
            pos=(line, column),
            encoding=TEXT_ENCODING.UTF8
          )
          newlines_consumed = value.count('\n')
          if newlines_consumed > 0:
            line += newlines_consumed
            column = 1
          else:
            column += len(value)
          logger.debug(f"Generated token: {token}")
          yield token
          index = match.end()
          break
      if match is None:
          raise ValueError(f"Unexpected character: {text[index]} at position: {(line, column)}")

### The AST Generator ###

class ParseError(Exception):
  pass

@dataclass
class TokenParser:
  """Parse the Tokens generated by the ChatLexer into an AST."""

  def parse(
    self,
    tokens: Generator[Token, None, None],
  ) -> Generator[Node, None, None]:
    while True:
      try:
        token = next(tokens)
        logger.debug(f"Received token: {token}")
      except StopIteration:
        break

      token_cache: list[Token] = []
      if token.kind == TOKEN_KIND.START_COMMENT:
        # start_comment_found = True
        eof_reached = False
        end_comment_found = False
        logger.debug("FOUND Start Comment; START Parsing Message")
        left_index = token.value.find(b'{')
        right_index = token.value.rfind(b'}')
        assert left_index != -1 and right_index != -1, f"Expected start comment to contain metadata, but got {token}"
        metadata = json.loads(token.value[left_index:right_index+1])

        # Consume tokens until two newlines are seen. If any of the tokens are not whitespace, raise an error
        newline_count = 0
        while not eof_reached:
          try:
            token = next(tokens)
            logger.debug(f"Received token: {token}")
          except StopIteration:
            eof_reached = True
            logger.info("EOF reached; STOP parsing message")
            break
          if token.kind == TOKEN_KIND.NEWLINE:
            newline_count += 1
            if newline_count == 2:
              break
            continue
          if token.kind != TOKEN_KIND.WHITESPACE:
            raise ParseError(f"Expected start comment to be followed by a newline, but got {token}")
        
        # Consume all tokens until an end comment is found
        while not eof_reached:
          try:
            token = next(tokens)
            logger.debug(f"Received token: {token}")
          except StopIteration:
            eof_reached = True
            logger.info("EOF reached; STOP parsing message")
            break
          if token.kind == TOKEN_KIND.END_COMMENT:
            end_comment_found = True
            logger.debug("FOUND End Comment; STOP parsing message")
            # Walk back the cache to trim the last 2 newlines & any proceeding whitespace
            newline_count = 0
            while len(token_cache) > 0:
              if token_cache[-1].kind == TOKEN_KIND.WHITESPACE:
                token_cache.pop()
              elif token_cache[-1].kind == TOKEN_KIND.NEWLINE:
                token_cache.pop()
                newline_count += 1
                if newline_count == 2:
                  break
                continue
              else:
                raise ParseError(f"End Comment can only be proceeded by whitespace or newline, but got {token_cache[-1]}")
            break
          
          token_cache.append(token)
        
        if len(token_cache) == 0:
          raise ParseError(f"Expected message to contain content, but got {token_cache}")
        if not end_comment_found:
          raise ParseError(f"Expected message to end with an end comment: {token_cache[-1].pos}")

        # Merge the token_cache into a single AST node
        logger.debug(f"Generating AST node from {len(token_cache)} tokens")
        ast_node: Node = MessageNode(
          metadata=MetadataNode(props=metadata),
          content=ContentNode(
            content=b''.join([token.value for token in token_cache]),
            encoding=token_cache[0].encoding.value,
            pos=token_cache[0].pos
          )
        )
        yield ast_node
      elif token.kind in {TOKEN_KIND.HORIZONTAL_RULE, TOKEN_KIND.WHITESPACE, TOKEN_KIND.NEWLINE} :
        logger.info("Skipping Token")
      else:
        logger.warning(f"Unexpected token '{token.kind}' at position '{token.pos}'")
        logger.info("Skipping Token")

def transpile_ast_to_message_dict(*nodes: Node) -> list[dict[str, str | dict[str, str]]]:
  msg_stack: list[dict[str, str | dict]] = []
  node_stack = list(filter(lambda n: isinstance(n, MessageNode), reversed(nodes)))
  node_count = len(node_stack)
  while len(node_stack) > 0:
    node = node_stack.pop()
    if isinstance(node, MessageNode):
      msg_stack.append({})
      node_stack.append(node.metadata)
      node_stack.append(node.content)
    elif isinstance(node, MetadataNode):
      assert "metadata" not in msg_stack[-1], "Metadata already exists in message object"
      assert isinstance(msg_stack[-1], dict), "Metadata must be a dict"
      msg_stack[-1]["metadata"] = node.props
    elif isinstance(node, ContentNode):
      assert "content" not in msg_stack[-1], "Content already exists in message object"
      assert isinstance(msg_stack[-1], dict), "Content must be a string"
      msg_stack[-1]["content"] = node.content.decode("utf-8")
    else:
      raise NotImplementedError(f"Node type {type(node)} not implemented")

  assert len(msg_stack) == node_count, "Message stack should be the same size as the number of nodes"
  return msg_stack

def transpile_message_dict_to_ast(*messages: dict[str, str | dict[str, str]]) -> list[Node]:
  # To reverse transpile we need to recalculate the line & column numbers
  ast_stack: list[Node] = []
  line = 1
  column = 1
  for msg in messages:
    assert "metadata" in msg, f"Message must contain metadata: {json.dumps(msg)}"
    assert "content" in msg, "Message must contain content"
    assert isinstance(msg["metadata"], dict), "Metadata must be a dict"
    assert isinstance(msg["content"], str), "Content must be a string"
    metadata = MetadataNode(props=msg["metadata"])
    content = ContentNode(
      content=msg["content"].encode("utf-8"),
      encoding=TEXT_ENCODING.UTF8.value,
      pos=(line, column)
    )
    line += msg["content"].count("\n")
    column = len(msg["content"].split("\n")[-1]) + 1
    ast_stack.append(MessageNode(metadata=metadata, content=content))
  return ast_stack

def transpile_ast_to_markdown(*nodes: Node) -> str:
  return "\n\n---\n\n".join([
    # The Message List
    "\n".join([
      # The Message
      f"<!-- start {json.dumps(node.metadata.props)} -->\n",
      node.content.content.decode(node.content.encoding),
      f"\n<!-- end -->"
    ]) for node in nodes if isinstance(node, MessageNode)
  ])

def _tests():
  dolphin_facts = "Dolphins are a widely distributed and diverse group of aquatic mammals. They are an informal grouping within the order Cetacea, excluding whales and porpoises, so to zoologists the grouping is paraphyletic. The dolphins comprise the extant families Delphinidae (the oceanic dolphins), Platanistidae (the Indian river dolphins), Iniidae (the new world river dolphins), and Pontoporiidae (the brackish dolphins). There are 40 extant species of dolphins. Dolphins range in size from the 1.7 m (5 ft 7 in) long and 50 kg (110 lb) Maui's dolphin to the 9.5 m (31 ft 2 in) and 10 t (11 short tons) killer whale. Several species exhibit sexual dimorphism. They have streamlined bodies and two limbs that are modified into flippers. Though not quite as flexible as seals, some dolphins can travel at 55.5 km/h (34.5 mph). Dolphins use their conical shaped teeth to capture fast moving prey. They have well-developed hearing which is adapted for both air and water and is so well developed that some can survive even if they are blind. Some species are well adapted for diving to great depths. They have a layer of fat, or blubber, under the skin to keep warm in the cold water."
  valid_markdown = \
f"""
<!-- start {{"role": "system"}} -->

You are a helpful assistant!

<!-- end -->

---

<!-- start {{"role": "user"}} -->

Can you tell me about Dolphins?

<!-- end -->

---

<!-- start {{"role": "assistant", "model": "gpt-4-0613"}} -->

{dolphin_facts}

<!-- end -->
""".strip()

  # The AST for the above markdown
  chat_ast: list[Node] = [
    MessageNode(
      metadata=MetadataNode(props={"role": "system"}),
      content=ContentNode(
        content=b"You are a helpful assistant!",
        encoding="utf-8",
        pos=(3, 1)
      )
    ),
    MessageNode(
      metadata=MetadataNode(props={"role": "user"}),
      content=ContentNode(
        content=b"Can you tell me about Dolphins?",
        encoding="utf-8",
        pos=(11, 1)
      )
    ),
    MessageNode(
      metadata=MetadataNode(props={"role": "assistant", "model": "gpt-4-0613"}),
      content=ContentNode(
        content=dolphin_facts.encode('utf-8'),
        encoding="utf-8",
        pos=(19, 1)
      )
    )
  ]

  message_objects: list[dict[str, str | dict]] = [
    {
      "metadata": {"role": "system"},
      "content": "You are a helpful assistant!"
    },
    {
      "metadata": {"role": "user"},
      "content": "Can you tell me about Dolphins?"
    },
    {
      "metadata": {"role": "assistant", "model": "gpt-4-0613"},
      "content": dolphin_facts
    }
  ]
  
  lexer = ChatLexer()
  parser = TokenParser()

  parsed_ast = list(parser.parse(
    lexer.tokenize(valid_markdown)
  ))
  logger.debug(f"Parsed Chat AST: {parsed_ast}")
  # Assert that the parsed ast is the same as the chat ast, if not then determine where the error is
  assert parsed_ast == chat_ast, next((f"Expected {expected}, got {actual} at position {i}" for i, (expected, actual) in enumerate(zip(chat_ast, parsed_ast)) if expected != actual), None)

  # Test that parsing valid_markdown produces the correct messages
  parsed_messages = transpile_ast_to_message_dict(*parsed_ast)
  assert tuple(parsed_messages) == tuple(message_objects), next((f"Expected {expected}, got {actual} at position {i}" for i, (expected, actual) in enumerate(zip(message_objects, parsed_messages)) if expected != actual), None)

  # # Test that generating markdown from the parsed messages produces valid_markdown
  generated_markdown = transpile_ast_to_markdown(*parsed_ast)
  # Assert the markdown matches, if not then determine where the error is
  if generated_markdown != valid_markdown:
    # Determine where the error is
    for i, (expected, actual) in enumerate(zip(valid_markdown, generated_markdown)):
      if expected != actual:
        # Find the line and column
        line = valid_markdown[:i].count("\n") + 1
        column = i - valid_markdown[:i].rfind("\n")
        # Print the line in the generated markdown that is wrong & then point out the character that is wrong
        correct_line = valid_markdown.split("\n")[line - 1]
        bad_line = generated_markdown.split("\n")[line - 1]
        carrot = " " * (column - 1) + "^"
        assert False, \
        f"""Markdown generation failed at line {line}, column {column}:
  Expected:
    {correct_line}
    {carrot}

  Got:
    {bad_line}
    {carrot}
"""
  logger.info(f"Generated Markdown...\n{generated_markdown}\n")
  logger.info("All tests passed!")

if __name__ == '__main__':
  logging.basicConfig(level=logging.DEBUG)
  try:
    _tests()
  except AssertionError as e:
    print(f"Test failed: {e}")
    exit(1)
  print("All tests passed!")
  exit(0)