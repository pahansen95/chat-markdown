"""# Response Module

Implement different response strategies for the LLM.

- [Single Shot](#single-shot)
- [Chain of Thought](#chain-of-thought)
- [Tree of Thoughts](#tree-of-thoughts)

## Single Shot

...

## Chain of Thought

...

## Tree of Thoughts

...

"""

from .interfaces import ChatMessage, LLM, Tokenizer
from typing import Any

import logging
import json
import re
logger = logging.getLogger(__name__)

class Error(Exception):
  ...

### Formatting ###

class FormattingError(Error):
  def __init__(
    self,
    language: str,
    reason: str,
    context: str | None = None,
    error: Exception | None = None
  ):
    """A Formatting Error
    Args:
      language (str): The Language we were trying to format to.
      reason (str): The reason the formatting failed.
      instructions (str): Instructions on how to fix the formatting.
      error (Exception | None, optional): The error that was raised. Defaults to None.
    """
    self.language = language.lower()
    self.reason = reason
    self.context = context
    self.error = error
  
  def __repr__(self) -> str:
    """Machine Friendly Representation"""
    return f"FormattingError(language={self.language}, reason={self.reason}, context={self.context}, error={self.error})"
  
  def __str__(self) -> str:
    """LLM Friendly Representation"""
    msg = f"""\
A Formatting Error was encountered as we attempted to format into {self.language}.

The following reason was given: {self.reason}
"""
    if self.context is not None:
      msg += f"""
The following context was provided: {self.context}
"""
    if self.error is not None:
      msg += f"""
The following error was raised: {self.error}
"""
    
    assert isinstance(msg, str)
    return msg

_json = dict[str, Any] | list[Any] | str | int | float | bool | None

async def json_from_natlang(
  content: str,
  top_level_type: str = "object",
) -> _json:
  """Attempt to extract JSON from natural language & deserialize it. If we fail to deserialize, raise an error detailing why.

  Args:
    content (str): The string to convert.
    top_level_type (str, optional): The type of the top level object. Defaults to "object". Possible values are ["object", "array", "string", "number", "boolean", "null"].

  Returns:
    _json: The deserialized JSON.
  
  Raises:
    FormattingError: If we cannot extract & deserialize JSON. A detailed explanation of the error is included.
  """

  # If the top level type is object or array, we should trim the string to the leading & trailing curly braces or brackets.

  if top_level_type in ["object", "array"]:
    l_char = "{" if top_level_type == "object" else "["
    r_char = "}" if top_level_type == "object" else "]"

    # Iterate through the string & identify all curly braces or brackets & their index
    char_stack: list[tuple[str, int]] = []
    for i, char in enumerate(content):
      last_char = content[i-1] if i > 0 else None
      if (char == l_char or char == r_char) and last_char != "\\":
        char_stack.append((
          char,
          i,
        ))
    
    pairs: list[tuple[int, int]] = []
    while len(char_stack) > 0:
      char, index = char_stack.pop() # Grab a curly brace or bracket
      if char == l_char: # If it's an opening curly brace or bracket
        # Add a new open pair
        pairs.append((index, -1))
      else: # If it's a closing curly brace or bracket
        # Find the matching opening one in the pairs list
        for i in range(len(pairs) - 1, -1, -1):
          if pairs[i][1] == -1:
            # Update the second element with the index of the closing one
            pairs[i] = (pairs[i][0], index)
            # Remove any nested pairs inside this pair
            for j in range(len(pairs) - 1, i, -1):
              if pairs[j][0] > pairs[i][0] and pairs[j][1] < pairs[i][1]:
                del pairs[j]
            break
        

    # Find the first opening curly brace or bracket.
    start_index: int = content.find(l_char)
    end_index: int = content.rfind(r_char)

    # JSON Parsing will fail so premptively raise a descriptive error
    if start_index == -1 or end_index == -1:
      reason = ""

      if start_index == -1:
        reason += f"no opening `{l_char}` was found"
      elif end_index == -1:
        if start_index == -1:
          reason += " and "
        reason += f"no closing `{r_char}` was found"    
      json_decode_error: Exception | None = None
      try:
        json.loads(content)
      except Exception as e:
        json_decode_error = e

      raise FormattingError(
        language="json",
        reason=reason,
        error=json_decode_error,
      )
    
    # Trim the string.
    content = content[start_index:end_index+1]

  # Attempt to convert the string to JSON.
  try:
    return json.loads(content) # type: ignore
  except Exception as e:
    raise FormattingError(
      language="json",
      reason=f"the string could not be converted to a JSON '{top_level_type}'",
      error=e,
    )

# async def attempt_to_fix(
#   content: str,
#   error: str,
#   llm: LLM,
#   tokenizer: Tokenizer,
#   attempts: int = 1
# ) -> tuple[str, ...]:
#   """Attempt to fix the formatting error using an LLM.
#   """
  
#   messages = [
#     ChatMessage(
#       content="Follow the user's instructions to the letter. Be precise, accurate & thorough.",
#       role="system",
#       model=None,
#       metadata={},
#     ),
#     ChatMessage(
#       content=f"""\
# I was trying to format some content but ran into an error:

# ```
# {str(error)}
# ```

# Can you help me fix the formatting?
# """,
#       role="user",
#       model=None,
#       metadata={},
#     ),
#     ChatMessage(
#       content=f"""\
# Sure, I can help. Please provide the content you were trying to format. I'll try to fix it.\
# """,
#       role="assistant",
#       model=None,
#       metadata={},
#     ),
#     ChatMessage(
#       content=content,
#       role="user",
#       model=None,
#       metadata={},
#     ),
#   ]

#   content_tokens_consumed = await tokenizer.tokens_consumed(messages[-1])

#   fix_attempts: ChatMessage | tuple[ChatMessage, ...] = await llm.chat(
#     messages=messages,
#     responses=attempts,
#     max_tokens=min(
#       int(content_tokens_consumed * 1.1), # Add ~10% padding to reduce out of token errors.
#       llm.context_window,
#     ),
#   )
#   if isinstance(fix_attempts, ChatMessage):
#     fix_attempts = (fix_attempts,)

#   return tuple(fix_attempt.content for fix_attempt in fix_attempts)


### Single Shot ###

async def single_shot(
  messages: list[ChatMessage],
  llm: LLM,
  **kwargs,
) -> tuple[ChatMessage, ...]:
  """Apply `Single Shot` to the prompt.
  """
  logger.debug("single_shot")
  return await llm.chat(messages, **kwargs)

### Chain of Thought ###

_cot_framework = """\
Chain of Thought (`CoT`) enhances LLM capabilities by crafting a more precise, accurate & objective response through decomposition of the orignal prompt.
The `CoT` framework is as follows:
1. Decompose the prompt into constituents.
2. Explicitly think through how to reply to each constituent. Favor novel ideas and avoid summarization.
3. Reply directly to the original prompt crafting a holistic response using the constituents as a guide.
"""

_cot_system_message = f"""\
Apply the Chain of Thought Framework in your responses.

{_cot_framework}

Your responses should follow this template. Anything in angle brackets is a placeholder and should be replaced with the appropriate value.

````markdown
1. **<Constituent Name>**
  - <the explicit thought process as an itemized list>
  - <...>
<n>. <...>

<A salient and concise response to the original prompt using the constituents as a guide...>
````
"""

_cot_decomposition_principles = """\
- Identify Thought Units: Break the prompt into distinct thought units that represent granular and self-contained units of thought.
- Simplify Complex Thought Units: Recursively breakout complex thought units while maintaining granularity, ensuring clarity and coherence.
- Organize Sequentially: Arrange thought units logically and sequentially to create a coherent flow of ideas.
- Ensure Clarity: Each thought unit should be clear, concise, and unambiguous, avoiding overlap or ambiguity within a single unit.
"""

_cot_decomposition_format = """\
Present the constituents as an itemized list. Each item should contain an identifying name and a natural language description of the constituent with no line breaks. The name & description should be seperated by a colon. There should be no sub items.

Exclusively use the following template; Anything in angle brackets is a placeholder and should be replaced with the appropriate value:
```markdown
1. <Identifying Constituent Name>: <Salient Description of the constituent as a single line.>

<n>. <Identifying Constituent Name>: <Salient Description of the constituent as a single line.>
```
"""

_cot_breakdown_system_message = f"""\
You will be given a prompt & relevant context. Your goal is to decompose the prompt into constituent sub-prompts using the provided principles & formatting. You are not to answer or reply to the prompt itself as your output will be used for further processing of the prompt.

Decomposition Principles:

{_cot_decomposition_principles}

Decomposition Format:

{_cot_decomposition_format}
"""

_cot_breakdown_constituent_regex = re.compile(
  r"(\d+)\.\s+([^:\n]+)(?::\s*.*?)?((?:(?!\n\d+\.)[\s\S])+)(?:\n(?=\d+\.)|$)",
  re.MULTILINE | re.DOTALL
)
"""
Matches items in a numbered markdown list.
Capture Groups:
  1. Item Number
  2. Name
  3. Description (Multiline)
"""

async def _cot_breakdown(
  messages: list[ChatMessage],
  llm: LLM,
  **kwargs,
) -> list[dict[str, str]]:
  """Breakdown the prompt into constituents using the llm."""
  logger.debug("_cot_breakdown")
  prompt_context = "\n\n".join([msg.content for msg in messages[-3:-1] if msg.role != 'system'])
  decomp_messages = [
    # System Message
    ChatMessage(
      role='system',
      content=_cot_breakdown_system_message,
      model=None,
      metadata={}
    ),
    ChatMessage(
      role='user',
      content="\n\n".join([
        "# Prompt to Decompose",
        f"````markdown\n{messages[-1].content.strip()}\n````",
        "# Prompt Context",
        f"````markdown\n{prompt_context.strip()}\n````",
      ]),
      model=None,
      metadata={}
    ),
    ChatMessage(
      role='assistant',
      content="I will now exclusively decompose the provided prompt into it's constituents adhering to the principles & formatting outlined in my system message. I will not respond or otherwise expand on the prompt. I must strictly adhere to the formatting & any provided templates.",
      model=None,
      metadata={}
    ),
  ]
  logger.debug("Message being submitted for Decomposition of the prompt:\n\n" + "\n\n".join([f"> {msg.role}\n\n{msg.content}" for msg in decomp_messages]))
  # raise NotImplementedError

  response: ChatMessage = (await llm.chat(
    messages=decomp_messages,
  ))[0]
  logger.debug(f"The Decomposition response was:\n{response.content}")

  # Extract the constituents from the response using the regex

  matches_found = _cot_breakdown_constituent_regex.findall(response.content)
  logger.debug("The Following Matches were found:\n" + "\n\n".join(f"{match=}" for match in matches_found))

  # ###
  # # for testing force the response to be misformatted
  # if matches_found:
  #   response.content = "\n\n".join([
  #     f"- **{match[1]}**\n  - {match[2]}"
  #     for match in matches_found
  #   ])
  #   logger.debug(f"Forcing the Decomposition response into bad formatting:\n{response.content}")
  #   matches_found = None
  # ###

  if not matches_found:
    # Reprompt the LLM instructing it to fix it's formatting
    fix_formatting_messages = [
      # System Message
      ChatMessage(
        role='system',
        content=f"Fix the formatting of the user message so it adheres to the following: {_cot_decomposition_format}",
        model=None,
        metadata={}
      ),
      # The Response
      ChatMessage(
        role='user',
        content=response.content,
        model=None,
        metadata={},
      ),
    ]
    logger.debug("Message being submitted for Fixing the formatting of the Decomposition response:\n\n" + "\n\n".join([f"> {msg.role}\n\n{msg.content}" for msg in fix_formatting_messages]))
    fixed_response: ChatMessage = (await llm.chat(
      messages=fix_formatting_messages,
    ))[0]
    logger.debug(f"The Fixed Formatting response was:\n{fixed_response.content}")
    matches_found = _cot_breakdown_constituent_regex.findall(fixed_response.content)
    logger.debug("The Following Matches were found:\n" + "\n\n".join(f"{match=}" for match in matches_found))

  # raise NotImplementedError

  constituents = [
    {
      "name": constituent[1],
      "description": constituent[2].replace("\n", " ") # Remove newlines from the description to fit it on one line
    }
    for constituent in sorted(
      matches_found,
      key=lambda t: t[0]
    )
  ]
  if len(constituents) == 0:
    logger.error("No constituents were generated. This is likely due to an output error w/ the LLM.")
    logger.info(f"# LLM Model Response\n\n{response.content}")
    raise RuntimeError("No constituents were generated. This is likely due to an output error w/ the LLM.")
  
  logger.debug(f"{constituents=}")

  return constituents

async def chain_of_thought(
  messages: list[ChatMessage],
  llm: LLM,
  cot_llm: LLM,
  **kwargs,
) -> tuple[ChatMessage, ...]:
  """Apply `Chain of Thought` (CoT) to the prompt.
  """
  logger.debug("chain_of_thought")
  logger.debug(f"{_cot_system_message=}")

  assert len(messages) > 0

  # Generate the constituents using the CoT LLM
  constituents = await _cot_breakdown(
    messages,
    cot_llm
  )
  assert len(constituents) > 0

  constituent_content = "\n".join([
    f"{index + 1}. {constituent['name']}: {constituent['description']}"
    for index, constituent in enumerate(constituents)
  ])

  cot_response: tuple[ChatMessage, ...] = await llm.chat(
    messages=[
      # System Message
      ChatMessage(
        role='system',
        content=f"{_cot_system_message}\n\n{next(m for m in messages if m.role == 'system').content}\n\nThese are the `CoT` constituents for your next response:\n\n{constituent_content}",
        model=None,
        metadata={}
      ),
      # # Constituents
      # ChatMessage(
      #   role='assistant',
      #   content=f"`CoT` constituents are:\n\n{constituent_content}",
      #   model=None,
      #   metadata={}
      # ),
      # Original Prompt stripping the System Message
      *[msg for msg in messages if msg.role != 'system'],
    ],
    **kwargs,
  )

  return tuple(
    ChatMessage(
      role='assistant',
      content=cot_response[i].content,
      model=cot_response[i].model,
      metadata={
        'chain_of_thought': {
          "constituents": constituents,
        }
      }
    )
    for i in range(len(cot_response))
  )

### Tree of Thoughts ###

async def tree_of_thoughts(
    
) -> ...:
  logger.debug("tree_of_thoughts")
  raise NotImplementedError
