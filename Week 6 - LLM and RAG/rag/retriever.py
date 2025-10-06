from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class Retriever:
    def __init__(self, documents):
        self.documents = documents
    
    def create_retriever(self):
        # Create and save to VectorDB
        vectorstore = Chroma.from_documents(documents=self.documents, 
                                            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
        return retriever