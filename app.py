from flask import Flask, render_template, request, jsonify
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

from transformers import pipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from src.prompt import get_prompt

app = Flask(__name__)

# -------------------------------
# ✅ Clean + shorten answer
# -------------------------------
def clean_text(text):
    text = re.sub(r'-\s+', '', text)          # fix broken words
    text = re.sub(r'\s+', ' ', text)          # remove extra spaces

    # keep only first 2–3 sentences
    sentences = text.split('.')
    short = '. '.join(sentences[:3]).strip()

    return short + '.' if short else text


# -------------------------------
# ✅ Load embeddings
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------
# ✅ Load FAISS index
# -------------------------------
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# 🔥 Better retrieval (important fix)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})


# -------------------------------
# ✅ Lightweight model (no crash)
# -------------------------------
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_length=512,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=pipe)


# -------------------------------
# ✅ Prompt + RAG chain
# -------------------------------
prompt = get_prompt()

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)


# -------------------------------
# ✅ Frontend route
# -------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------------
# ✅ Chat API
# -------------------------------
@app.route("/get", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message")

        if not user_input:
            return jsonify({"answer": "Please enter a question."})

        result = rag_chain.invoke({"input": user_input})

        answer = clean_text(result["answer"])

        print("\nUSER:", user_input)
        print("BOT:", answer)

        return jsonify({"answer": answer})

    except Exception as e:
        print("ERROR:", str(e))
        return jsonify({"answer": "Error occurred. Check terminal."})


# -------------------------------
# ✅ Run app
# -------------------------------
if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))