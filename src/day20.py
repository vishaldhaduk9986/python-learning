import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import OpenAI

# -----------------------------
# 1. Load and preprocess the PDF
# -----------------------------
pdf_path = "example.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

loader = PyPDFLoader(pdf_path)
pages = loader.load()

# -----------------------------
# 2. Split text into chunks
# -----------------------------
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# -----------------------------
# 3. Create embeddings and vector store
# -----------------------------
embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
vector_store = FAISS.from_documents(docs, embeddings)

# -----------------------------
# 4. Create retriever
# -----------------------------
retriever = vector_store.as_retriever()

# -----------------------------
# 5. Initialize LLM
# -----------------------------
llm = OpenAI(
    temperature=0,
    openai_api_key=os.environ.get("OPENAI_API_KEY")
)

# -----------------------------
# 6. Define the summarization prompt
# -----------------------------
prompt = PromptTemplate.from_template(
    "Summarize the following content clearly and concisely:\n\n{context}"
)

# -----------------------------
# 7. Create the chain
# -----------------------------
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

# -----------------------------
# 8. Run a query (must use 'input', not 'query')
# -----------------------------
query = "What is the main topic of the document?"

# Note: create_retrieval_chain expects input key = 'input'
result = qa_chain.invoke({
    "input": query,  # ✅ fixed key
    "return_source_documents": True
})

# -----------------------------
# 9. Display the results
# -----------------------------
print("\n=== Query ===")
print(query)

print("\n=== Answer ===")
print(result.get("answer") or result.get("result") or "No answer returned.")


