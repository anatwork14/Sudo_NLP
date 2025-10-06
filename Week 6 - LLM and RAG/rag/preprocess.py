# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Preprocessor:
    def __init__(self, documents):
        self.documents = documents
    
    def split_documents(self):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300, 
            chunk_overlap=50)
        return text_splitter.split_documents(self.documents)