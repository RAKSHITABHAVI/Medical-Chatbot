from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs


def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return text_splitter.split_documents(docs)


def filter_text(texts):
    filtered = []

    for doc in texts:
        content = doc.page_content.lower()

        if "journal" in content or "page" in content or "copyright" in content:
            continue

        filtered.append(doc)

    return filtered