**Document Search and Summarization System**

An end-to-end Document Search and Summarization application built using Information Retrieval (IR) techniques, cosine similarity, and Streamlit.
The system retrieves the most relevant documents for a user query and generates an extractive summary of the results.

This project is designed to be lightweight, stable, and cloud-deployable, while still covering all core requirements of a Retrieval-Augmented workflow.

**🚀 Project Overview**

The goal of this project is to build a system that:

Loads and preprocesses a document corpus

Performs semantic-style document search

Retrieves the top-K most relevant documents

Generates a concise summary of the retrieved content

Provides a simple and interactive user interface

Can be deployed on Streamlit Cloud without dependency issues

Due to cloud resource constraints, the system uses TF-IDF + cosine similarity as a reliable and explainable retrieval baseline.

🧠 System Architecture
User Query
   ↓
TF-IDF Vectorization
   ↓
Cosine Similarity
   ↓
Top-K Relevant Documents
   ↓
Extractive Summarization
   ↓
Streamlit UI Output

**🛠️ Tech Stack**

Python

Streamlit – User Interface

Scikit-learn – TF-IDF & cosine similarity

NumPy – Numerical operations

(No external APIs, no paid services)

**📁 Project Structure**

document-search-project/
│

├── app.py        # Backend logic (search + summarization)

├── rag_st.py     # Streamlit UI

├── data/         # Document corpus (.txt files)

│   ├── doc1.txt
│   ├── doc2.txt
│   └── doc3.txt
├── requirements.txt
└── README.md

⚙️ How the System Works
1️⃣ Data Preparation

Documents are loaded from the data/ directory

Text is cleaned (lowercasing, trimming whitespace)

**2️⃣ Document Search**

TF-IDF is used to convert documents into vectors

User query is vectorized using the same TF-IDF model

Cosine similarity is computed between query and documents

Top-K most relevant documents are selected

**3️⃣ Summarization**

Extractive summarization is applied

The first N important sentences from retrieved documents are used

Summary length is configurable from the UI

**4️⃣ User Interface**

Built using Streamlit

Allows users to:

Enter a query

Choose number of documents (Top-K)

Choose summary length

Handles empty input and missing documents gracefully

**▶️ How to Run the Application**
1️⃣ Install dependencies
pip install -r requirements.txt

2️⃣ Run the app
streamlit run rag_st.py


The app will open automatically in your browser.

**✨ Features**

🔍 Semantic-style document search

🧠 Extractive summarization

🎛️ Adjustable Top-K retrieval

📏 Adjustable summary length

⚠️ Graceful handling of empty input and missing data

☁️ Fully deployable on Streamlit Cloud

**📌 Evaluation Strategy**

Retrieval Evaluation:
Accuracy@K by checking whether relevant documents appear in top-K results.

Summarization Evaluation:
Manual evaluation based on relevance, coverage, and coherence.
