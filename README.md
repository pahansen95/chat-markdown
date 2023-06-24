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

  --personality=PERSONALITY_ID       Select a personality preset for AI chat responses.
                                     Available presets: creative, balanced, reserved.
                                     (Default: balanced)

  --temperature=FLOAT_VALUE          Set a custom temperature value for the AI chat responses.
                                     (Overwrites the personality preset temperature value.)

  --tokens=INT_VALUE                 Set a custom maximum token count for the AI chat responses.
                                     (Overwrites the personality preset token value.)

  --topp=FLOAT_VALUE                 Set a custom top_p value for the AI chat responses.
                                     (Overwrites the personality preset top_p value.)
  
  --reverse                          Reverse the Roles of the AI and the User in the Chat.

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
```

## Markdown Chat Format

> Note: This is currently a WIP. Formatting is a bit rigid.

- Wrap Messages w/ a `<!--- start {} --->` & `<!--- stop --->` HTML Comment
  - Pad your messages with a newline between the comments
- Seperate Messages with a line break: `---`
  - Pad line breaks with a newline before & after them

Synatx Example:
```text
<!-- start {"role": "system"} -->

You are a highly experienced marine biologist. You are participating as part of a QA. Answer questions precisely & accurately. Keep your prose concise & your language simple.

<!-- end -->

---

<!-- start {"role": "user"} -->

Can you tell me about Dolphins?

<!-- end -->
```