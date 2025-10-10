"""
day16.py
---------
Demonstrates chaining two LLM prompts using LangChain's SequentialChain.

Developer notes:
- Expects `OPENAI_API_KEY` in the environment to run the live example.
- This file is an example script that runs immediately when executed.
    In production code prefer composing chains inside functions and adding
    tests that inject a fake LLM.
"""

import os
from langchain_openai import OpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

# Instantiate an LLM using the OpenAI wrapper. The API key is pulled
# from the environment to avoid hardcoding secrets in source.
llm = OpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))

summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text briefly:\n\n{text}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

keyword_prompt = PromptTemplate(
    input_variables=["summary"],
    template="Extract keywords from this summary, separated by commas:\n\n{summary}"
)
keyword_chain = LLMChain(llm=llm, prompt=keyword_prompt, output_key="keywords")

combined_chain = SequentialChain(
    chains=[summary_chain, keyword_chain],
    input_variables=["text"],
    output_variables=["summary", "keywords"]
)

input_text = "LangChain is an open-source framework that simplifies building applications with large language models by managing prompts, chains, agents, and integrations."

result = combined_chain({"text": input_text})

print("Summary:", result["summary"])
print("Keywords:", result["keywords"])
