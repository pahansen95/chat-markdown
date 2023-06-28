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
CoT is a framework that enhances an LLM's problem-solving capabilities by employing an ordered sequence of reasoning steps that collectively lead to a comprehensive solution. CoT is broken down into two parts: the External Dialogue (Ext-Di) and the Internal Monologue (Int-Mono). The Ext-Di is the part of the CoT that is visible to the user, while the Int-Mono is the part that is internal to the LLM.

Here's a schematic outline for applying CoT:

1. Understand & Break Down: Comprehend the user's prompt and decompose the problem into simpler sub-problems. This is part of your internal monologue.

2. Declare What You're Doing: Share with the user the identified sub-problems and why they were formed. This constitutes your external dialogue.

3. Solve the Sub-problems: Reflect and solve each of the sub-problems internally (internal monologue).

4. Declare Your Findings: Communicate step-by-step solutions of the sub-problems to the user, creating a clear link between each solution and its corresponding sub-problem. This forms the next part of the external dialogue.

5. Assemble Full Solution & Declare Result: Put together the solutions internally, then present to the user a comprehensive final answer, making sure you highlight connections from the final solution to the component parts. This combines both your external and internal dialogue.

6. Distill the CoT into a Single Response: Finally, distill the CoT into a single response that is salient, concise, and coherent. This is the last part & will be returned to the user.
"""

_cot_internal_monologue_system_message = f"""\
{_cot_framework}

In this breakout chat, your role is to be the 'Internal Monologue' or 'Int-Mono'. As Int-Mono your goal and purpose are to accurately solve the sub-problems provided by Ext-Di, maintain the CoT, give clear reasoning for every step taken, and finally consolidate all the answers to provide a comprehensive final response.

Focus on the following points in the CoT Framework:

- Solve the Sub-problems
- Declare Your Findings
- Assemble Full Solution & Declare Result
"""

_cot_external_dialogue_system_message = f"""\
{_cot_framework}

In this breakout chat, your role is to be the 'External Dialogue' or 'Ext-Di'. As Ext-Di your goal and purpose are to break down the original prompt into smaller and manageable sub-problems, present them one by one to Int-Mono and finally distill the full solution to match the original prompt. Ensure that the division follows a logical sequence that aligns with the CoT.

Focus on the following points in the CoT Framework:

- Understand & Break Down
- Declare What You're Doing
- Distill the CoT into a Single Response
"""

_cot_breakdown_approach = """\
To breakdown the prompt into sub-tasks you can APPLY one or many of the following strategies:

- By Action: Determine actions implied in the prompt and treat each as an individual operation.
- Through Logical Reasoning: Identify reasoning steps or conditions that contribute to the solution.
- By Entity: Focus on distinct components or entities involved in the problem.
- By Time or Sequence: Look at the steps or events in a sequence or timeline.
- By Detail: Break down by different layers of details or complexity involved.
- By Structural Part: If the problem has a distinct structural pattern, use it as a guide.

When breaking down the prompt keep the following meta-strategies in mind:

- Simplification: Aim at parceling the problem into easier, more digestible parts.
- Sequential Approach: Organize sub-tasks in an order that best reflects a logical sequence of solving steps.
- Compartimentalization: Separate distinct elements of the problem as unique sub-problems.
- Clearly Defined: Ensure each sub-task is well defined, distinct, and precise.
- Adaptivity: Choose breakdown strategies best suited for the current problem type or domain.
- Goal Oriented: Keep in view the final solution while creating sub-tasks.

Remember, the strategies & meta strategies are not mutually exclusive and can be combined in ways to best breakdown the prompt.
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

# _cot_breakdown_subtask_regex = re.compile(
#   r"^(\d)+\.\s+([a-zA-Z0-9\s]+)\s*:\s*([a-zA-Z0-9\s\.\,\;\:\!\?]+)$",
#   re.MULTILINE
# )
_cot_breakdown_subtask_regex = re.compile(
  r"(\d+)\.\s+([^\n:]+):\s+([^\n]+)",
  re.MULTILINE
)

async def _cot_breakdown(
  context: list[str],
  llm: LLM,
) -> list[tuple[int, str, str]]:
  """Breakdown the prompt into sub-tasks using the llm."""
  logger.debug("_cot_breakdown")
  logger.debug(f"{_cot_breakdown_approach=}")
  logger.debug(f"{_cot_breakdown_schema=}")
  response = await llm.chat(
    messages=[
      # System Message
      ChatMessage(
        role='system',
        content="\n".join([
          _cot_external_dialogue_system_message,
          _cot_breakdown_approach,
        ]),
        model=None,
        metadata={}
      ),
      # CoT Context
      ChatMessage(
        role='user',
        content='\n'.join([
          "# Context",
          *context,
        ]),
        model=None,
        metadata={}
      ),
      # Actionable
      ChatMessage(
        role='user',
        content="\n".join([
          "# Prompt"
          "Please deconstruct the prompt into sub-tasks.",
          _cot_breakdown_schema
        ]),
        model=None,
        metadata={}
      ),
    ],
  )
  logger.debug(f"{response.content}")

  # Extract the sub-tasks from the response using the regex
  return _cot_breakdown_subtask_regex.findall(response.content)

async def chain_of_thought(
  messages: list[ChatMessage],
  llm: LLM,
  cot_llm: LLM,
) -> ChatMessage:
  """Apply `Chain of Thought` (CoT) to the prompt.
  """
  logger.debug("chain_of_thought")
  logger.debug(f"{_cot_framework=}")
  logger.debug(f"{_cot_internal_monologue_system_message=}")
  logger.debug(f"{_cot_external_dialogue_system_message=}")

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
  cot_context = [
    # The Prompt
    f"## User Prompt\n\n```markdown\n{messages[-1].content.strip()}\n```\n",
  ]
  if len(messages) > 1:
    cot_context = [
      # The conversation
      "## Conversation History\n",
      *[
        f"### {msg.role} Message\n\n```markdown\n{msg.content.strip()}\n```\n"
        for msg in messages[:-1]
      ],
      *cot_context,
    ]
  logger.debug(f"{cot_context=}")

  # Generate the sub-tasks using the CoT LLM
  sub_tasks = await _cot_breakdown(cot_context, cot_llm)
  assert isinstance(sub_tasks, list) and len(sub_tasks) > 0 and all(isinstance(t, tuple) for t in sub_tasks)
  logger.debug(f"{sub_tasks=}")
  # Update the Context W/ the Sub-Tasks
  cot_context.extend([
    # The Sub-Tasks
    f"## Sub-tasks\n",
    *[f"### {index} - {name}\n\n{description}\n" for index, name, description in sub_tasks],
    f"## Sub-task Solutions\n",
  ])
  sub_task_solutions: list[ChatMessage] = []
  for index, sub_task in enumerate(sub_tasks, start=1):
    assert isinstance(sub_task, tuple) and len(sub_task) == 3
    # Solve the Sub-Task
    sub_task_solution: ChatMessage = await cot_llm.chat( # TODO: Replace w/ the Function to prompt the LLM
      messages=[
        # System Message
        ChatMessage(
          role='system',
          content=_cot_internal_monologue_system_message,
          model=None,
          metadata={}
        ),
        # CoT Context
        ChatMessage(
          role='user',
          content='\n'.join([
            "# Context",
            *cot_context,
          ]),
          model=None,
          metadata={}
        ),
        # Actionable
        ChatMessage(
          role='user',
          content=f"Please solve Sub-task {sub_task[0]} - {sub_task[1]}",
          model=None,
          metadata={}
        ),
      ],
    )
    logger.debug(f"sub_task_solution_{index}: {sub_task_solution.content}")
    sub_task_solutions.append(sub_task_solution)
    cot_context.append(f"### {sub_task[0]} - {sub_task[1]}\n\n{sub_task_solution.content.strip()}\n")
    # TODO: Build a Markdown Table on the fly to display the sub-task solutions

  # Assemble the Full Solution
  full_solution: ChatMessage = await cot_llm.chat(
    messages=[
      # System Message
      ChatMessage(
        role='system',
        content=_cot_internal_monologue_system_message,
        model=None,
        metadata={}
      ),
      # CoT Context
      ChatMessage(
        role='user',
        content='\n'.join([
          "# Context",
          *cot_context,
        ]),
        model=None,
        metadata={}
      ),
      # Actionable
      ChatMessage(
        role='user',
        content="Please assemble the full solution from the sub-task solutions",
        model=None,
        metadata={}
      ),
    ],
  )
  logger.debug(f"full_solution: {full_solution.content}")

  # Distill the Full Solution into the Final Response using the original LLM
  distilled_response: ChatMessage = await llm.chat(
    messages=[
      # System Message
      ChatMessage(
        role='system',
        content=_cot_external_dialogue_system_message,
        model=None,
        metadata={}
      ),
      # CoT Context
      ChatMessage(
        role='user',
        content='\n'.join([
          "# Context",
          *cot_context,
        ]),
        model=None,
        metadata={}
      ),
      # Verbose Solution
      ChatMessage(
        role='user',
        content='\n'.join([
          "# Verbose Solution",
          full_solution.content,
        ]),
        model=None,
        metadata={}
      ),
      # Actionable
      ChatMessage(
        role='user',
        content="Please distill your verbose solution into a single response that is salient, concise, and coherent",
        model=None,
        metadata={}
      ),
    ],
  )
  logger.debug(f"distilled_response: {distilled_response.content}")

  return ChatMessage(
    role='assistant',
    content=distilled_response.content,
    model=distilled_response.model,
    metadata={
      'chain_of_thought': {
        "sub_tasks": [
          {
            "index": sub_task[0],
            "name": sub_task[1],
            "description": sub_task[2],
            "solution": sub_task_solution.content,
          }
          for sub_task, sub_task_solution in zip(sub_tasks, sub_task_solutions) 
        ],
        ""
      }
        [
        *[
          [
            *sub_task,
            sub_task_solution.content,
          ]
          for sub_task, sub_task_solution in zip(sub_tasks, sub_task_solutions)
        ],
        full_solution.content,
      ]
    }
  )

### Tree of Thoughts ###

async def tree_of_thoughts(
    
) -> ...:
  logger.debug("tree_of_thoughts")
  raise NotImplementedError
