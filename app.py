import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# Page config
st.set_page_config(page_title="Medical Chatbot")

st.title("🩺 Medical Chatbot (AI Powered)")

# Load FAISS DB
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

# Load LLM
@st.cache_resource
def load_model():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )
    return pipe

# Initialize
db = load_db()
llm = load_model()

# User input
query = st.text_input("Ask a medical question:")

if query:
    docs = db.similarity_search(query, k=3)

    # Combine context
    context = " ".join([doc.page_content for doc in docs])

    # FINAL PROMPT (clean + structured)
    prompt = f"""
You are a helpful medical assistant.

Use ONLY the information from the context below to answer the question.

Context:
{context}

Question:
{query}

Instructions:
- Give a short, clear, and structured answer
- Avoid repeating unnecessary text
- Explain in simple words

Answer:
"""

    # Generate answer
    result = llm(prompt, max_length=200)

    st.write("### Answer:")
    st.write(result[0]["generated_text"])