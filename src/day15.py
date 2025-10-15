"""
day15.py
---------
Small example demonstrating how to call an OpenAI LLM via LangChain.

Developer notes:
- This module is intentionally small and demonstrates a single-run example
    (not intended to run as a long-lived server).
- It tries to import `OpenAI` from a few compatible LangChain packages so
    it works across different local setups. If none are installed, the
    module will fail fast with an explanatory message.
- The OpenAI API key is read from the `OPENAI_API_KEY` environment variable.
    For local development: export OPENAI_API_KEY=your_key

Usage:
        python src/day15.py

Returned values / exit codes:
- 0 : success
- 1 : OPENAI_API_KEY missing

"""

import sys
from typing import Optional

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from src.utils import get_openai_api_key, make_chat_llm


def main() -> int:
    api_key = get_openai_api_key()
    if not api_key:
        print(
            "OPENAI_API_KEY is not set. To run this example, set the environment variable:\n"
            "  export OPENAI_API_KEY=your_api_key_here\n\n"
            "You can get a key from https://platform.openai.com/ or run with a different LLM provider.",
            file=sys.stderr,
        )
        return 1

    # Initialize ChatOpenAI via shared helper. Returns None if unavailable.
    llm = make_chat_llm(openai_api_key=api_key, model_name="gpt-4o")
    if llm is None:
        print("Failed to initialize ChatOpenAI; check installation and API key.", file=sys.stderr)
        return 1

    # Prompt template to politely rewrite a sentence
    prompt_template = PromptTemplate(
        input_variables=["sentence"],
        template="Rewrite the following sentence politely:\n\n{sentence}",
    )

    # Create chain with the LLM and prompt template
    chain = LLMChain(llm=llm, prompt=prompt_template)

    sentence = "Give me the report now."
    polite_sentence = chain.run(sentence)

    print("Original:", sentence)
    print("Polite:", polite_sentence)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
