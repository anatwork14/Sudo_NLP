from langchain.prompts import ChatPromptTemplate
from retriever import Retriever
from preprocess import Preprocessor
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from dataloader import DataLoader

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

class Agent:
    def __init__(self, datasource: str):
        loader = DataLoader(datasource)         # ✅ create DataLoader object
        documents = loader.load_data()          # ✅ call instance method
        preprocessor = Preprocessor(documents=documents)
        
        retrie_instance = Retriever(preprocessor.split_documents())
        self.retriever = retrie_instance.create_retriever()

    def run(self, userprompt: str):
        docs = self.retriever.get_relevant_documents(userprompt)
        context = "\n\n".join([doc.page_content for doc in docs])

        template = """Answer the question based on the context below:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm

        return chain.invoke({"context": context, "question": userprompt})
