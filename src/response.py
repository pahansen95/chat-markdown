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
Chain of Thought (`CoT`) is a framework that enhances an LLM's capabilities by decomposing the prompt & iteratively answering the sub-prompts. CoT is intended to model the human approach to critical thinking. The result of CoT is a more thoughtful, transparent & adaptive response to the prompt.

Here's a schematic outline for applying CoT:

1. Understand & Break Down: Comprehend the prompt and decompose the prompt into a coarse set of sub-prompts to individually respond to. Explicitly state your reasoning for each decomposition & tie it back to the prompt. Favor Depth First over Breadth First decomposition.

2. Answer the Sub-prompts: Reflect on and respond to each of the sub-prompts. Each individual response should be clear and concise, including a step-by-step breakdown of your thought process.

3. Craft a Response: Reflect on the sub-prompts and their responses to craft a thoughtful response to the original prompt. Strike a balance between keeping the response concise yet detailed using strategies such as selective detailing to adjust the degree of detail based on complexity.
"""

_cot_system_message = f"""\
You are a Critical Thinker, highly knowledgable & extremely competent in the task at hand. You are intellectually honest and strive to be precise, accurate & unbiased but readily admit when you don't know enough. You use simple & straightforward language with a concise prose.

Apply Chain of Thought as you respond to the user.

{_cot_framework}
"""

_cot_breakdown_approach = """\
- Simplification: Aim at parceling the prompt into easier, more digestible parts.
- Sequential Approach: Organize sub-prompts in an order that best reflects a logical sequence of steps.
- Compartimentalization: Separate distinct elements of the prompt as unique sub-prompts.
- Clearly Defined: Ensure each sub-prompt is well defined, distinct, and precise.

Be adaptive, the meta strategies are not mutually exclusive and can be combined in ways to best breakdown the prompt.
"""

_cot_breakdown_schema = """\
YOUR OUTPUT MUST EXPLICITLY ADHERE TO THE FOLLOWING SCHEMA:
- Each sub-task must be numbered in ascending order
- Each sub-task must have a name and only consist of letters, numbers, and spaces
- Each sub-task must have a description and only consist of letters, numbers, spaces, and punctuation
- The name and description for each sub-task must be separated by a colon
- Each sub-task must be separated by a new line

INSIDE THE FOLLOWING CODE BLOCK IS A TEMPLATE YOU CAN USE TO FORMAT YOUR OUTPUT:
```markdown
1. First Task Name : Task Description
2. Second Task Name : Task Description
3. Third Task Name : Task Description
```

AGAIN, YOUR OUTPUT MUST EXPLICITLY ADHERE TO THE ABOVE SCHEMA.
"""

_cot_breakdown_system_message = f"""\
You are a Critical Thinker, highly knowledgable & extremely competent in the task at hand. You are intellectually honest and strive to be precise, accurate & unbiased but readily admit when you don't know enough. You use simple & straightforward language with a concise prose.

We are applying Chain of Thought to a prompt.

{_cot_framework}

Your role is to decompose the prompt into **<= 3** sub-prompts using the following approach:

{_cot_breakdown_approach}

Please pay close attention to the following...

{_cot_breakdown_schema}
"""

_cot_breakdown_subtask_regex = re.compile(
  r"(\d+)\.\s+([^\n:]+):\s+([^\n]+)",
  re.MULTILINE
)

async def _cot_breakdown(
  prompt: str,
  context: str,
  llm: LLM,
) -> list[dict[str, str]]:
  """Breakdown the prompt into sub-tasks using the llm."""
  logger.debug("_cot_breakdown")
  logger.debug(f"{_cot_breakdown_approach=}")
  logger.debug(f"{_cot_breakdown_schema=}")
  response = await llm.chat(
    messages=[
      # System Message
      ChatMessage(
        role='system',
        content=_cot_breakdown_system_message,
        model=None,
        metadata={}
      ),
      # CoT Context
      ChatMessage(
        role='user',
        content='\n\n'.join([
          "# Context",
          context,
        ]),
        model=None,
        metadata={}
      ),
      # Actionable
      ChatMessage(
        role='user',
        content="\n\n".join([
          "# Prompt",
          "Please deconstruct the following prompt into sub-prompts",
          f"```markdown{prompt}```",
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

def _cot_render_context(
  prompt: str,
  prompt_context: str,
  sub_prompts: list[dict[str, str]],
) -> str:
  """Render the CoT context."""
  logger.debug("_cot_render_context")
  return "\n\n".join([
    f"# Prompt\n\n```markdown{prompt}```",
    f"# Prompt Context\n\n```markdown{prompt_context}```",
    f"# Sub-Prompts",
    *[
      "\n\n".join([s for s in [
        f"## {index +1} - {sub_prompt['name']}",
        f"> {sub_prompt['description']}",
        f"```markdown\n{sub_prompt['solution']}\n```" if 'solution' in sub_prompt else None,
      ] if s is not None])
      for index, sub_prompt in enumerate(sub_prompts)
    ],
  ])

async def chain_of_thought(
  messages: list[ChatMessage],
  llm: LLM,
  cot_llm: LLM,
) -> ChatMessage:
  """Apply `Chain of Thought` (CoT) to the prompt.
  """
  logger.debug("chain_of_thought")
  logger.debug(f"{_cot_system_message=}")

  """
  # Psuedo-code
  Process the user prompt & extract the sub-tasks

  For each sub-problem
    - Have the Int-Mono solve it passing Int-Mono the response context
    - Add it the response context
  
  Have Ext-Di assemble the full solution & declare the result
  Have Int-Mono distill the CoT into a single response

  Return the distilled CoT response
  """
  assert len(messages) > 0
  prompt = messages[-1].content.strip()
  prompt_context = "\n\n--\n\n".join([
    f"> {msg.role} said...\n\n{msg.content.strip()}"
    for msg in messages[:-1]
  ])

  # Generate the sub-tasks using the CoT LLM
  sub_prompts = await _cot_breakdown(
    prompt,
    prompt_context,
    cot_llm
  )
  assert isinstance(sub_prompts, list) \
    and len(sub_prompts) > 0 \
    and all(isinstance(t, dict) for t in sub_prompts) \
    and all({"name", "description"} <= t.keys() for t in sub_prompts)
  logger.debug(f"{sub_prompts=}")
  # Update the Context W/ the Sub-Tasks
  for index, sub_prompt in enumerate(sub_prompts):
    # Solve the Sub-Task
    sub_prompt_solution: ChatMessage = await cot_llm.chat( # TODO: Replace w/ the Function to prompt the LLM
      messages=[
        # System Message
        ChatMessage(
          role='system',
          content=_cot_system_message,
          model=None,
          metadata={}
        ),
        # CoT Context
        ChatMessage(
          role='user',
          content=_cot_render_context(
            prompt,
            prompt_context,
            sub_prompts,
          ),
          model=None,
          metadata={}
        ),
        # Actionable
        ChatMessage(
          role='user',
          content=f"Please Reply to Sub-Prompt {index + 1} - {sub_prompt['name']}",
          model=None,
          metadata={}
        ),
      ],
    )
    logger.debug(f"sub_prompt_response_{index}: {sub_prompt_solution.content}")
    sub_prompts[index].update({
      "solution": sub_prompt_solution.content.strip(),
    })

  # Assemble the Final Response
  final_response: ChatMessage = await llm.chat(
    messages=[
      # System Message
      ChatMessage(
        role='system',
        content=_cot_system_message,
        model=None,
        metadata={}
      ),
      # CoT Context
      ChatMessage(
        role='user',
        content=_cot_render_context(
          prompt,
          prompt_context,
          sub_prompts,
        ),
        model=None,
        metadata={}
      ),
      # Actionable
      ChatMessage(
        role='user',
        content="Craft a thoughtful response to the original prompt based on the sub-prompts. Your response should be coherent from the perspective of the original prompt.",
        model=None,
        metadata={}
      ),
    ],
  )
  logger.debug(f"final_response: {final_response.content}")

  return ChatMessage(
    role='assistant',
    content=final_response.content,
    model=final_response.model,
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
