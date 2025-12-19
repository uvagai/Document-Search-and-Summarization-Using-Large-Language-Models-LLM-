import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------- LOAD DOCUMENTS ----------
def load_documents(data_dir="data"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, data_dir)

    texts = []

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data folder not found: {data_path}")

    for file in os.listdir(data_path):
        if file.endswith(".txt"):
            with open(os.path.join(data_path, file), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)

    return texts


# ---------- BUILD VECTOR STORE ----------
def build_vectorstore(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.create_documents(texts)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# ---------- RAG PIPELINE (CLOUD SAFE) ----------
def run_rag_pipeline(query, top_k=3):
    texts = load_documents("data")
    vectorstore = build_vectorstore(texts)

    docs = vectorstore.similarity_search(query, k=top_k)

    answer = "\n\n".join([doc.page_content[:300] for doc in docs])

    return answer, docs
