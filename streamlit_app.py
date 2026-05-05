import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="Medical Chatbot")

st.title("🩺 Medical Chatbot")

# Load embeddings
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 3})

# Input
query = st.text_input("Ask your question:")

# Output
if query:
    docs = retriever.get_relevant_documents(query)

    if docs:
        answer = " ".join([doc.page_content for doc in docs[:2]])
    else:
        answer = "Sorry, I don't know."

    st.write("### Answer:")
    st.write(answer)