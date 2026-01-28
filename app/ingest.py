from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter

with open("data/docs.txt") as f:
    text = f.read()

splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.create_documents([text])

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

vectorstore.save_local("vectorstore")
print("✅ Vector store built")
