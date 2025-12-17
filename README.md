# Document-Search-and-Summarization-Using-Large-Language-Models-LLM-
Document Search & Summarization using RAG

A Retrieval-Augmented Generation (RAG) application built using LangChain, local transformer models, and Streamlit.
The system retrieves relevant documents from a local knowledge base and generates concise summaries for user queries.

**🚀 Project Overview**

This project demonstrates an end-to-end RAG pipeline where:

Documents are loaded from local text files

Text is embedded using SentenceTransformers

Semantic search is performed using FAISS

Retrieved context is summarized using a local transformer model

The pipeline is exposed through an interactive Streamlit UI

✅ No OpenAI API

✅ No paid services

✅ Fully local & offline

✅ Windows compatible


**🧠 Architecture**
User Query
   ↓
Retriever (FAISS + Embeddings)
   ↓
Relevant Documents
   ↓
Local LLM (Summarization)
   ↓
Final Answer

**🛠️ Tech Stack**

Python

LangChain

SentenceTransformers

FAISS

Hugging Face Transformers

Streamlit

📁** Project Structure**

everquint_rag_project/

│
├── app.py        # Backend: RAG logic using LangChain

├── rag_st.py     # Frontend: Streamlit UI

├── data/         # Knowledge base (text documents)

│   ├── doc1.txt

│   ├── doc2.txt

│   └── doc3.txt

└── README.md

**⚙️ Installation & Setup**

1️⃣ Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd everquint_rag_project

2️⃣ Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install streamlit langchain langchain-community
pip install sentence-transformers transformers faiss-cpu

▶️ How to Run the Application
streamlit run rag_st.py


The app will open automatically in your browser.

**✨ Features**

🔍 Semantic document retrieval

🧠 Context-aware summarization

⚡ Fast FAISS-based search

🖥️ Simple and clean UI

📦 Modular backend-frontend design

📌 Example Use Cases

Knowledge base search

Document summarization

Study notes generation

Internal document Q&A
