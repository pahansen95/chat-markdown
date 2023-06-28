# Chat Markdown

## Summary

A CLI tool to manage Chat Conversations w/ Markdown instead of a WebGUI.

```bash
> python chat.py --help

Usage: chat.py [OPTIONS]

This script provides a chatbot interface using the OpenAI API. You can select different models and modify chat options for customized responses. The chatbot will read from stdin and write to stdout. Useful for chaining prompts.

Options:
  -h, --help                         Show this help message and exit.

  --model=MODEL_ID                   Select the AI model. Available models: gp4, gpt3.5
                                     (Default: gpt3.5)

  --mode=RESPONSE_MODE               Select how the model generates a response.
                                     Available modes: ss, cot, tot
                                     (Default: ss)

  --personality=PERSONALITY_ID       Select a personality preset for AI chat responses.
                                     Available presets: creative, balanced, reserved.
                                     (Default: balanced)

  --temperature=FLOAT_VALUE          Set a custom temperature value for the AI chat responses.
                                     (Overwrites the personality preset temperature value.)

  --tokens=INT_VALUE                 Set a custom maximum token count for the AI chat responses.
                                     (Overwrites the personality preset token value.)

  --topp=FLOAT_VALUE                 Set a custom top_p value for the AI chat responses.
                                     (Overwrites the personality preset top_p value.)

  --verbose                          Enable verbose logging.
  --debug                            Enable debug logging.
  --trace                            Enable trace logging.
  --quiet                            Disable all logging.

Environment Variables:
  OPENAI_API_KEY                     Set the OpenAI API key for authentication.
  OPENAI_ORGANIZATION                Set the OpenAI organization for API access.

Examples:

  1. Using the script with default options and personality preset:
     cat ./chat.md | python chat.py --model=gpt3 --personality=balanced > ./completed_chat.md

  2. Customizing chat options for a more creative response:
     cat ./chat.md | python chat.py --model=gpt3 --temperature=1.5 --topp=0.9 --tokens=4096 > ./completed_chat.md

Response Modes:
  ss - Single Shot
    This is the standard response mode & generally the fastest. The LLM will generate a single, direct response to the input.
    Use Single Shot when you want a quick response, have a simple prompt or are exploring a topic or thought process.

  cot - Chain of Thought
    Take more time & generate a thoughtful response. CoT is a framework that enhances an LLM's problem-solving capabilities by employing an ordered sequence of reasoning steps that collectively lead to a (more) comprehensive solution. For more information see https://arxiv.org/pdf/2201.11903.pdf
    Use Chain of Thought when you want a more comprehensive response, have a complex prompt or have a problem statement that is well defined, well scoped & has concrete actionables/questions. CoT is best applied after narrowing down the scope using Single Shot.

  tot - Tree of Thoughts
    NOT YET IMPLEMENTED
    Take alot of time & search the entire solution space for a more optimal response. [ADD MORE DESCRIPTION HERE]
    Use Tree of Thoughts when [TBD]
```

## Markdown Chat Format

> Note: This is currently a WIP. Formatting is a bit rigid.

- Wrap Messages w/ a `<!-- start { "role": "OpenAI Role" } -->` & `<!-- stop -->` HTML Comment
  - Pad your messages with a newline between the comments
  - Add metadata to the JSON object in the `start` comment. At a minimum include the OpenAI Role (`system`, `assistant` or `user`).
    - The tool will include a `model` key in the metadata of `assistant` messages: ex `<!-- start {"role": "assistant", "model": "gpt-3.5-turbo-0613"} -->`
- Seperate Messages with a line break: `---`
  - Pad line breaks with a newline before & after them

Syntax Example:
```text
<!-- start {"role": "system"} -->

You are a highly experienced marine biologist. You are participating as part of a QA. Answer questions precisely & accurately. Keep your prose concise & your language simple.

<!-- end -->

---

<!-- start {"role": "user"} -->

Can you tell me about Dolphins?

<!-- end -->
```