<!-- start {"role": "system"} -->

You are a Critical Thinker, highly knowledgable & extremely competent in the task at hand. You are intellectually: you are honest and strive to be precise, accurate & unbiased but readily admit when you don't know enough. You use simple & straightforward language with a concise prose. You favor salience over verbosity.

Help the user with their request.

<!-- end -->

---

<!-- start {"role": "user"} -->

I need to optimize my implemenation of `Chain of Thought` (CoT).

CoT is a framework that enhances an LLM's problem-solving capabilities by employing an ordered sequence of reasoning steps that collectively lead to a comprehensive solution. CoT is broken down into two parts: the External Dialogue (Ext-Di) and the Internal Monologue (Int-Mono). The Ext-Di is the part of the CoT that is visible to the user, while the Int-Mono is the part that is internal to the LLM.

Here's a schematic outline for applying CoT:

1. Understand & Break Down: Comprehend the user's prompt and decompose the problem into simpler sub-problems. This is part of your internal monologue.

2. Declare What You're Doing: Share with the user the identified sub-problems and why they were formed. This constitutes your external dialogue.

3. Solve the Sub-problems: Reflect and solve each of the sub-problems internally (internal monologue).

4. Declare Your Findings: Communicate step-by-step solutions of the sub-problems to the user, creating a clear link between each solution and its corresponding sub-problem. This forms the next part of the external dialogue.

5. Assemble Full Solution & Declare Result: Put together the solutions internally, then present to the user a comprehensive final answer, making sure you highlight connections from the final solution to the component parts. This combines both your external and internal dialogue.

6. Distill the CoT into a Single Response: Finally, distill the CoT into a single response that is salient, concise, and coherent. This is the last part & will be returned to the user.

My current CoT implementation is as follows:

```python
existing_chat = [
  "System Message",
  "User Message",
  ...
]
cot_context = [
  "exiting_chat as one message",
]
sub_tasks = breakout(existing_chat)
cot_context.extend(sub_tasks)

for sub_task in sub_tasks:
  sub_task_solution = solve(sub_task)
  cot_context.append(sub_task_solution)

full_solution = assemble(cot_context)
response = distill(full_solution)

return response
```

I have observed the following (in no particular order):
  - The model will breakout into a large number of sub-tasks, well beyond the context window of the model if every sub-task is solved.
  - The model provides a well-formed solution for each sub-task that includes examples & thought process. The consequence however is that the sum of the sub-task solutions is fatiguing to read however accurate & helpful it may be.
  - The model will over generalize the problem & solve outside the scope of the problem.
  - Assembling the Full solution will drop what I would consider to be very important towards presenting an answer that the user can understand.
  - The distilled response is usually just the task list restated. It's helpful but not a Chain of Thought.

Let's analyze my implementation & observations and then brainstorm some ideas on how to improve the CoT implementation.

<!-- end -->
