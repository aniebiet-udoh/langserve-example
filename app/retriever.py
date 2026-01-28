from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def get_retriever():
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("vectorstore", embeddings)
    return db.as_retriever()
