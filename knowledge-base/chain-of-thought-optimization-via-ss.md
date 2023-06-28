<!-- start {"role": "system"} -->

You are a Critical Thinker, highly knowledgable & extremely competent in the task at hand. You are intellectually: you are honest and strive to be precise, accurate & unbiased but readily admit when you don't know enough. You use simple & straightforward language with a concise prose. You favor salience over verbosity.

Help the user with their request.

<!-- end -->

---

<!-- start {"role": "user"} -->

I need to optimize my implementation of `Chain of Thought` (CoT).

Chain of Thought (`CoT`) is a framework that enhances an LLM's capabilities by decomposing the prompt & iteratively answering the sub-prompts. CoT is intended to model the human approach to critical thinking. The result of CoT is a more thoughtful, transparent & adaptive response to the prompt.

> # A Framework for Apply Chain of Thought
>
> 1. Understand & Break Down: Comprehend the prompt and decompose the prompt into a coarse set of sub-prompts to individually respond to. Explicitly state your reasoning for each decomposition & tie it back to the prompt. Favor Depth First over Breadth First decomposition.
> 2. Answer the Sub-prompts: Reflect on and respond to each of the sub-prompts. Each individual response should be clear and concise, including a step-by-step breakdown of your thought process.
> 3. Craft a Response: Reflect on the sub-prompts and their responses to craft a thoughtful response to the original prompt. Strike a balance between keeping the response concise yet detailed using strategies such as selective detailing to adjust the degree of detail based on complexity.

My current CoT implementation follows:

````python
from ._interfaces import ChatMessage, LLM

import logging
import re
logger = logging.getLogger(__name__)

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

def _cot_render_context(
  prompt: str,
  prompt_context: str,
  sub_prompts: list[dict[str, str]],
) -> str:
  """Render the CoT context."""
  logger.debug("_cot_render_context")
  return "\n\n".join([
    f"# Prompt\n\n```markdown\n{prompt}\n```",
    f"# Prompt Context\n\n```markdown\n{prompt_context}\n```",
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
````

First, provide a comprehensive analysis of my CoT Framework & my Python implementation of CoT.

Next, identify computational improvements in the source code and cognitive improvements in the Framework.

<!-- end -->

---

<!-- start {"role": "assistant", "model": "gpt-4-0613", "mode": "Single Shot"} -->

# Analysis of the Framework

The Chain of Thought (CoT) Framework, in principle, seems robust and presents a methodical approach to problem-solving. It's mirroring the human process of critical thinking, including breaking down problems, reflecting on individual parts, and then formulating a coherent answer.

Its components have been detailed clearly:

- **Understanding & Breakdown:** Advocating for a strategic breakdown of the prompt into 3 sub-prompts is advantageous. It allows for efficient problem management and underscores the depth-first approach, which is usually beneficial for intricate problems.

- **Answer Sub-prompts:** The requirement to produce clear, concise, and logical steps in answering each subtask assists in producing a transparent and understandable solution.

- **Craft Response:** Combining the above into a concise, yet comprehensive, reply ensures the user question was fully understood and addressed.

# Analysis of the Python Implementation

Your implementation appears to capture the framework well. You generate the sub-tasks using the CoT approach and solve each one independently. After that, you combine the derived responses into a final response that can be returned.

You also use relevant assertions to ensure certain conditions are met throughout the process. These assertions can be a good defense against unexpected inputs or execution errors, aiding in debugging and maintaining the program.

However, the complexity resulting from the asynchronous nature of the tasks, usage of regular expressions, and nesting of multiple helper functions within the main `chain_of_thought` function does increase the cognitive load for any person trying to understand the codebase quickly.

# Computational and Cognitive Improvements

**Computational Improvements:**

- **Code Duplication:** The program contains significant duplicated code, especially related to system messages and CoT framework rendering. It might be beneficial to create a helper function to handle this.

- **Regular Expressions:** These are CPU-intensive, especially for large input strings. If inputs are always well-formatted, a simpler parser could improve performance.

- **Asynchronous Calls:** While asynchronous calls are parallelizable and beneficial, they're best for I/O-bound operations. Here, it may lead to additional overhead without clear advantages.

**Cognitive Improvements:**

- **Error Handling:** Adding an explicit error handling routine would enhance readability and ease debugging of the application.

- **Comments and Documentation:** While you have made use of comments to explain the pseudo-code, additional comments throughout the application would be beneficial.

- **Code Modularity:** Final solution crafting can be moved to another function that can be individually tested and developed. This would improve your code's maintainability.

- **Simplifying Workflows:** The workflow feels slightly overly complicated. Simplifying it, perhaps by reducing the amount of different steps or merging similar ones, could make the work easier.

These improvements could lead to a more maintainable and comprehensible codebase, improving the overall quality of the application and potentially speeding up its execution.

<!-- end -->

---

<!-- start {"role": "user"} -->

Could you provide an in depth exposé on your suggested cognitive improvements.

<!-- end -->

---

<!-- start {"role": "assistant", "model": "gpt-4-0613", "mode": "Single Shot"} -->

Sure, I'd be happy to break down the cognitive improvements a bit more.

1. **Explicit Error Handling:**
    - Python uses exceptions for error handling. Exceptions provide a clean way to catch any unexpected behaviour and indicate where exactly the problem is.
    - When you use assertions, it suggests you're testing for internal inconsistencies that should never occur. This might appear to be a testing tactic, not a runtime defensive tactic.
    - Applying explicit error handling would allow you to give detailed error messages to indicate exactly what is wrong. This could make debugging easier by letting you know what to fix.

2. **Comments and Documentation:**
    - Well-written code is self-documenting according to the adage, but well-placed comments provide context which code can't.
    - Comments that explain the *why* behind your code, any assumptions made, or the purpose of functions will make it easier for others to understand your thought process, and maintain or enhance your code.
    - Docstrings for function descriptions and their inputs and outputs can assist others in understanding what each function is supposed to do. This also allows for automatic documentation generation.

3. **Code Modularity:**
    - Breaking down complex tasks into smaller, simpler functions improves readability and makes the code more maintainable.
    - Testing becomes manageable since unit tests can be written to test each separate function.
    - In the context of coding, the Single Responsibility Principle (SRP) infers that a function should do only one thing. This is a basic building block for maintainable code.
    - It is suggested to move the crafting of the final response to its own function which can be individually developed, tested, and code-reviewed.

4. **Simplify Workflows:**
    - Workflows should be as simple as possible; this makes them easier to understand, test, and troubleshoot.
    - If workflows feel overly complex, this suggests that they may have been "over-engineered". Over-engineering refers to designs that are more complex than necessary.
    - Given the nature of how workflows are used in development, simple workflows are quicker to run and easier to manage.
    - Identifying areas where similar steps are repeated and combining them or reducing the number of steps where possible can make the process much easier.

These cognitive improvements all aim at the same goal: enhancing the readability, maintainability, and usability of your code.

<!-- end -->
