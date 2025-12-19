import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------- LOAD DOCUMENTS ----------
def load_documents(data_dir="data"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_dir)

    documents = []

    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    documents.append(Document(page_content=text))

    return documents


# ---------- BUILD VECTOR STORE ----------
def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ---------- RAG PIPELINE (NO TRANSFORMERS) ----------
def run_rag_pipeline(query, top_k=3):
    documents = load_documents("data")
    vectorstore = build_vectorstore(documents)

    docs = vectorstore.similarity_search(query, k=top_k)

    # Simple extractive answer
    answer = "\n\n".join([doc.page_content[:300] for doc in docs])

    return answer, docs
