from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import PromptTemplate

from src.utils import make_chat_llm

# Instantiate an LLM using the shared helper so tests can monkeypatch
# make_chat_llm to return a dummy LLM.
llm = make_chat_llm(model_name="gpt-4o", temperature=0)
if llm is None:
    raise RuntimeError("ChatOpenAI is not available or OPENAI_API_KEY is missing")

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
