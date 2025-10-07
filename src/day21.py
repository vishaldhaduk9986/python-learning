from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import os

app = FastAPI(
    title="PDF Q&A API",
    description="Upload a PDF and ask questions about its content using LangChain + OpenAI.",
    version="1.0.0"
)

# Global variable to store embeddings (you can enhance this with per-user sessions later)
vector_store = None


# -----------------------------
# 1️⃣ Upload PDF and Create Embeddings
# -----------------------------
@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF, split it into chunks, and store embeddings in FAISS.
    """
    global vector_store

    # Save uploaded PDF temporarily
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        # Load PDF and split into chunks
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.split_documents(docs)

        # Create embeddings
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise HTTPException(status_code=500, detail="Missing OpenAI API key.")

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Build FAISS vector store
        vector_store = FAISS.from_documents(split_docs, embeddings)

        return {"msg": f"PDF '{file.filename}' uploaded successfully and indexed."}

    finally:
        # Always remove temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -----------------------------
# 2️⃣ Question Model
# -----------------------------
class QuestionReq(BaseModel):
    question: str


# -----------------------------
# 3️⃣ Ask a Question about the Uploaded PDF
# -----------------------------
@app.post("/ask")
async def ask(question_req: QuestionReq):
    """
    Answer a user's question using the uploaded PDF.
    Must be called after /upload_pdf.
    """
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet. Please upload a file first.")

    # Initialize retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # get top 3 relevant chunks

    # Initialize OpenAI LLM
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(status_code=500, detail="Missing OpenAI API key.")

    llm = OpenAI(
        temperature=0,
        openai_api_key=openai_api_key
    )

    # Build retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    # Run question against document context
    result = qa_chain({"query": question_req.question})

    # Extract answer and sources
    answer = result.get("result") or "No answer found in the document."
    sources = result.get("source_documents", [])

    # Prepare output
    return {
        "question": question_req.question,
        "answer": answer,
        "sources": [
            {"page": doc.metadata.get("page", "N/A"), "content": doc.page_content[:300] + "..."}
            for doc in sources
        ]
    }
