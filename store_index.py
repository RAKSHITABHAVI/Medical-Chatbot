from src.helper import load_pdf, split_text, filter_text
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_index():
    docs = load_pdf("data/Medical_book.pdf")
    texts = split_text(docs)
    filtered = filter_text(texts)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(filtered, embeddings)
    vectorstore.save_local("faiss_index")

    print("Index created!")

if __name__ == "__main__":
    create_index()