from langchain.vectorstores import Chroma

def create_vector_store(documents, embeddings):
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="vectorstore/chroma_db"
    )
    return vectordb