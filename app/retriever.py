from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader


def get_doc_loader(file_path: str):
    loader = TextLoader(file_path=file_path)
    documents = loader.load()
    return documents


def get_web_loader(url: str):
    loader = WebBaseLoader(url)
    webdocs = loader.load()
    return webdocs


def get_retriever():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore", embeddings)
    return db.as_retriever()
