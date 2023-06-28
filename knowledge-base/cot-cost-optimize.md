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

<!-- start {"role": "assistant", "model": "gpt-4-0613", "mode": "Chain of Thought", "chain_of_thought": {"sub_prompts": [{"name": "Identify Costly Components ", "description": "Review the current implementation to identify the components or operations that require the most computational resources."}, {"name": "Optimize LLM Usage ", "description": "Propose strategies to reduce the frequency or necessity of LLM usage in the program."}, {"name": "Suggest Structural Changes ", "description": "Provide ideas for restructuring the code or implementation to make it more cost-efficient without compromising the functionality."}]}} -->

1. **Identify Costly Components**

The `_cot_breakdown` and `chain_of_thought` functions seem to be the most computationally expensive. Both these functions call the `llm.chat` method which is likely to be the most resource-intensive operation. Within the `_cot_breakdown` method, the input message to the `llm.chat` method is a concatenation of previous messages and a new user message. The length of these messages is proportional to the computational cost.

2. **Optimize LLM Usage**

Optimize the usage of `llm.chat` by reducing the number of times it is called and the amount of data sent per call. Since the token cost for gpt-3.5-turbo is primarily dependent on the input and output tokens, reducing them would directly lower costs.

- `Chain of Thought (CoT) breakdown`: Since the breakdown subtask is using a separate instance of LLM (cot_llm), consider using a lighter LLM model or a regular expression-based breakup method for the breakdown task. This operation doesn't necessarily require a full-powered LLM.
- `Reducing input tokens`: Tokens in the input prompt can be reduced by sending shorter system and user messages or by limiting the number of past conversation turns sent. Investigate whether the context information is crucial for generating the correct output or if it can be reduced without losing accuracy.

3. **Suggest Structural Changes**

- `Caching`: Introduce caching to avoid repeated computations. If a prompt is repeated, you can store the result of the prompt and return the result from the cache instead of calling `llm.chat` again.
- `Parallel processing`: If applicable and the use case allows, some of the LLM inquiries related to the sub-prompts or different chain of thought processes can be made concurrently.
- `Asynchronous execution`: In the current implementation, each LLM chat call is awaited immediately after being called. If these calls can be made asynchronous, e.g., by using Python's `asyncio.gather()`, it could allow for other tasks to run simultaneously, thus potentially decreasing the execution time.

<!-- end -->

---

<!-- start {"role": "user"} -->

<!-- end -->