"""
day17.py
---------
Small example showing how to load a PDF and summarize its text using
LangChain's PDF loader and an LLM.

Developer notes:
- This script expects `example.pdf` to exist in the repo root. It's a
    minimal demo that loads, concatenates the first two pages, and runs a
    summarization chain.
- For testability, break this flow into functions and inject a fake
    loader/LLM rather than calling them at import time.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import os


# Load the PDF (replace with a real path if running locally)
loader = PyPDFLoader("example.pdf")
pages = loader.load_and_split()

# Combine both pages' text for a short demo summary
text = "\n".join([page.page_content for page in pages[:2]])

# Summarize with OpenAI via LangChain using the API key from env
llm = ChatOpenAI(
    temperature=0,
    openai_api_key=os.environ.get("OPENAI_API_KEY"),
    model_name="gpt-4o"  # Latest OpenAI model as of October 2025
)
prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following document in 2-3 sentences:\n\n{text}"
)
chain = LLMChain(llm=llm, prompt=prompt)
summary = chain.run(text)

print("Document Summary:")
print(summary)
