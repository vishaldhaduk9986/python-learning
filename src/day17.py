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
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from src.utils import make_chat_llm


def summarize_pdf(path: str):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()

    text = "\n".join([page.page_content for page in pages[:2]])

    llm = make_chat_llm(model_name="gpt-4o", temperature=0)
    if llm is None:
        raise RuntimeError("ChatOpenAI is not available or OPENAI_API_KEY missing")

    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following document in 2-3 sentences:\n\n{text}",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(text)


if __name__ == "__main__":
    s = summarize_pdf("example.pdf")
    print("Document Summary:")
    print(s)
