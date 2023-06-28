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

from ._interfaces import ChatMessage, LLM

import logging
import re
logger = logging.getLogger(__name__)

### Single Shot ###

async def single_shot(
  messages: list[ChatMessage],
  llm: LLM,
) -> ChatMessage:
  """Apply `Single Shot` to the prompt.
  """
  logger.debug("single_shot")
  return await llm.chat(messages)

### Chain of Thought ###

_cot_framework = """\
Chain of Thought (`CoT`) enhances LLM capabilities by breaking the prompt down into sub-prompts, analyzing them and building a thoughtful response. The steps are:
1. Breakup the prompt into three consituent sub-prompts.
2. Answer each individual sub-prompt, in order, providing detail proportinal to it's complexity.
3. Craft a holistic response to the original prompt reflecting on the sub-prompts and their responses.
"""

_cot_system_message = f"""\
Apply Chain of Thought as you respond to the user.

{_cot_framework}
"""

_cot_breakdown_system_message = f"""\
Deconstruct the given prompt into 3 sub-prompts using these principles:
- Simplify: Break the prompt into manageable parts.
- Sequential Approach: Keep the logical progression in mind.
- Compartimentalization: Treat distinct elements as unique sub-prompts.
- Clarity: Each sub-prompt should be defined precisely.

Finally, follow this schema:
- Format sub-tasks as an ordered markdown list.
- Each Sub-task item has a 3 word name and salient one-line description separated by a colon.

Example:
```markdown
1. First Task Name : Short task description on a single line.
```
"""

_cot_breakdown_subtask_regex = re.compile(
  r"(\d+)\.\s+([^\n:]+):\s+([^\n]+)",
  re.MULTILINE
)

async def _cot_breakdown(
  messages: list[ChatMessage],
  llm: LLM,
) -> list[dict[str, str]]:
  """Breakdown the prompt into sub-tasks using the llm."""
  logger.debug("_cot_breakdown")
  response = await llm.chat(
    messages=[
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
          "# Context",
          *[m.content for m in messages[-3:-1]],
          "# Prompt",
          messages[-1].content,
        ]),
        model=None,
        metadata={}
      ),
    ],
  )
  logger.debug(f"{response.content}")

  # Extract the sub-tasks from the response using the regex
  return [
    {
      "name": sub_prompt[1],
      "description": sub_prompt[2],
    }
    for sub_prompt in sorted(
      _cot_breakdown_subtask_regex.findall(response.content),
      key=lambda t: t[0]
    )
  ]

async def chain_of_thought(
  messages: list[ChatMessage],
  llm: LLM,
  cot_llm: LLM,
) -> ChatMessage:
  """Apply `Chain of Thought` (CoT) to the prompt.
  """
  logger.debug("chain_of_thought")
  logger.debug(f"{_cot_system_message=}")

  assert len(messages) > 0

  # Generate the sub-tasks using the CoT LLM
  sub_prompts = await _cot_breakdown(
    messages,
    cot_llm
  )
  logger.debug(f"{sub_prompts=}")

  sub_prompt_content = "\n".join([
    f"{index + 1}. {sub_prompt['name']}: {sub_prompt['description']}"
    for index, sub_prompt in enumerate(sub_prompts)
  ])

  cot_response = await llm.chat(
    messages=[
      # System Message
      ChatMessage(
        role='system',
        content="\n\n".join([
          _cot_system_message,
          "Original System Message was:",
          "````markdown",
          next(m for m in messages if m.role == 'system').content,
          "````",
        ]),
        model=None,
        metadata={}
      ),
      # Actionable
      ChatMessage(
        role='user',
        content=f"`CoT` Sub-Prompts are:\n\n{sub_prompt_content}",
        model=None,
        metadata={}
      ),
      # Original Prompt stripping the System Message
      *[msg for msg in messages if msg.role != 'system'],
    ],
  )

  return ChatMessage(
    role='assistant',
    content=cot_response.content,
    model=cot_response.model,
    metadata={
      'chain_of_thought': {
        "sub_prompts": sub_prompts,
      }
    }
  )

### Tree of Thoughts ###

async def tree_of_thoughts(
    
) -> ...:
  logger.debug("tree_of_thoughts")
  raise NotImplementedError
