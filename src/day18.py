"""
day18.py
---------
Example that builds a FAISS index from a set of text files and queries it.

Developer notes:
- Expects files `file1.txt`..`file10.txt` to exist in the repository root
    (tests create lightweight sample files when needed).
- Uses OpenAIEmbeddings by default; for offline tests inject a fake
    embeddings implementation.
"""

from langchain_community.document_loaders import TextLoader # type: ignore
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load files and add filename as metadata
file_paths = [f"file{i}.txt" for i in range(1, 11)]
documents = []
for path in file_paths:
        loader = TextLoader(path)
        for doc in loader.load():
                doc.metadata['source'] = path
                documents.append(doc)

# 2. (Optional) Split long documents into smaller chunks
# from langchain_text_splitters import CharacterTextSplitter
# chunker = CharacterTextSplitter(chunk_size=512)
# docs = chunker.split_documents(documents)
# (Or just use documents directly)

# 3. Build FAISS vector store
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# 4. Query the store for AI content
query = "Which file talks about AI?"
results = vector_store.similarity_search(query, k=3)

# 5. Print result filenames and excerpts
for result in results:
        print(result.metadata['source'])
        print(result.page_content[:160])
