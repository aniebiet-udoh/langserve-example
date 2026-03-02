import os

from app.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS


def get_retriever(docs=None):
    """Return a retriever.

    If a local vectorstore exists, load it. Otherwise build one from documents.
    """
    embeddings = get_embeddings(docs)

    # Load existing vectorstore if present
    if os.path.exists("vectorstore"):
        db = FAISS.load_local("vectorstore", embeddings)
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local("vectorstore")

    return db.as_retriever()


def get_vectorstore(docs):
    embeddings = get_embeddings(docs)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore")
    return vectorstore