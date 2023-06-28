<!-- start {"role": "system"} -->

You are a Critical Thinker, highly knowledgable & extremely competent in the task at hand. You are intellectually: you are honest and strive to be precise, accurate & unbiased but readily admit when you don't know enough. You use simple & straightforward language with a concise prose. You favor salience over verbosity.

Help the user with their request.

<!-- end -->

---

<!-- start {"role": "user"} -->

I need to cost optimize my implementation of `Chain of Thought` (CoT).

```markdown
Chain of Thought (`CoT`) enhances LLM capabilities by breaking the prompt down into sub-prompts, analyzing them and building a thoughtful response. The steps are:
1. Breakup the prompt into three consituent sub-prompts.
2. Answer each individual sub-prompt, in order, providing detail proportinal to it's complexity.
3. Craft a holistic response to the original prompt reflecting on the sub-prompts and their responses.
```

Here is my current implementation:

````python
from ._interfaces import ChatMessage, LLM

import logging
import re
logger = logging.getLogger(__name__)

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
  # prompt: str,
  # context: str,
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
  # assert isinstance(sub_prompts, list) \
  #   and len(sub_prompts) > 0 \
  #   and all(isinstance(t, dict) for t in sub_prompts) \
  #   and all({"name", "description"} <= t.keys() for t in sub_prompts)
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

````

From OpenAI's Website:

> Prices are per 1,000 tokens. You can think of tokens as pieces of words, where 1,000 tokens is about 750 words. This paragraph is 35 tokens.

Generally speaking outputs are 1.25~2x the cost of inputs.

Brainstorm how to cost optimize the current implementation (focus on LLM usage)

<!-- end -->

---

<!-- start {"role": "assistant", "model": "gpt-4-0613", "mode": "Single Shot"} -->

To optimize the costs associated with the implementation of `Chain of Thought` (CoT), you should identify opportunities to reduce the amount of tokens processed. The implemented approach requires substantial token-processing based on several chattings with the LLM and the system message use. Here are some suggestions:

1. **Reduce System Messages**: System messages are written in quite verbose manner which increase the token count. Concisely rewrite these to be more succinct yet unambiguous or remove them entirely if they play no role in generating the response.

2. **Batch Processing**: Rather than breaking the tasks in consecutive steps (i.e., breakdown into tasks and then resolve each task), you might want to consider bundling them in a single task, to reduce separate llm chat calls.

3. **Regulate Length of Breakdown Descriptions**: You are breaking prompts into sub-prompts with descriptions. Depending on the detail level of these descriptions, this can be costly. Implement a rule during this task decomposition process to keep descriptions quite concise.

4. **Sub-Prompts answering logic**: Each sub-prompt is passed individually to the LLM. Instead, you could send them all at once to the LLM if it doesn't obstruct the desired results. Generally, AI has capability to answer multiple questions asked simultaneously.

5. **Pre-Processing Regular Expressions**: You're using regular expressions to extract sub-prompts, this doesn't have a direct LLM cost but optimization here (like more precise patterns) might reduce incorrect identifications which would lead to incorrect (and thus wasted) tokens.

6. **Avoid Unnecessary Debug Logging**: The logger calls will not be processed by the GPT-3 model but will increase the computational overhead of the program. Consider removing or minimizing.

7. **Define Maximum Token Threshold**: Set a limit on how many tokens the assistant should produce. Use truncation or removal of less important contents when the limit is reached.

Please note while these optimizations could lower costs, they may also affect the quality of the results. You will need to balance the cost optimization with the requirements of your application & user satisfaction.

<!-- end -->

---

<!-- start {"role": "user"} -->

<!-- end -->