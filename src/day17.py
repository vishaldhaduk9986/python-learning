from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

import os


# Load the 2-page PDF
loader = PyPDFLoader("example.pdf")   # Replace with your document path
pages = loader.load_and_split()

# Combine both pages' text
text = "\n".join([page.page_content for page in pages[:2]])

# Summarize with OpenAI via LangChain
llm = OpenAI(temperature=0,openai_api_key=os.environ.get("OPENAI_API_KEY"))
prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following document in 2-3 sentences:\n\n{text}"
)
chain = LLMChain(llm=llm, prompt=prompt)
summary = chain.run(text)

print("Document Summary:")
print(summary)
