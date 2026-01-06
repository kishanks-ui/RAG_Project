import streamlit as st
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

st.title("RAG Web App")

@st.cache_resource
def initialize_rag():
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    return RAGSearch(store)

rag_search = initialize_rag()

query = st.text_input("Ask a question")

if query:
    summary = rag_search.search_and_summarize(query, top_k=3)
    st.subheader("Answer")
    st.write(summary)
