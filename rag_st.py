import streamlit as st
from app import run_rag_pipeline

st.set_page_config(
    page_title="Document Search & Summarization",
    layout="wide"
)

st.title("📚 Document Search & Summarization (RAG)")
st.write("Semantic document retrieval with extractive summarization")

query = st.text_input("🔎 Enter your query")

col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Top-K Documents", 1, 5, 3)
with col2:
    summary_len = st.slider("Summary Length (sentences)", 1, 10, 5)

if st.button("Search & Summarize"):
    # Case 1: No user input
    if not query.strip():
        st.warning("⚠️ There is no data in your document.")
    else:
        with st.spinner("Processing..."):
            summary, docs = run_rag_pipeline(query, top_k, summary_len)

        # Case 2: No documents available
        if summary is None:
            st.warning("⚠️ There is no data in your document.")
        else:
            st.subheader("📝 Summary")
            st.write(summary)

            st.subheader("📄 Retrieved Documents")
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Document {i}**")
                st.write(doc[:500] + "...")
