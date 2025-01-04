import warnings
from langchain_google_genai import GoogleGenerativeAI
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


warnings.filterwarnings("ignore")
dotenv.load_dotenv()

api_key=os.environ["GOOGLE_API_KEY"]


llm = GoogleGenerativeAI(model="gemini-pro",api_key=api_key)





model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    
    

)

def create_vector_database():
   loader = CSVLoader(file_path="medquad.csv", source_column="question", encoding="utf-8")
   data=loader.load()
   vectorDB=FAISS.from_documents(documents=data,embedding=hf)
   vectorDB.save_local("FAISS_DB")


def get_chain():
   vectorDB=FAISS.load_local("FAISS_DB",embeddings=hf,allow_dangerous_deserialization=True)
   retriever=vectorDB.as_retriever()
   prompt_template = """Given the following context and a question, generate an answer based on this context. 
Try to include as much relevant information as possible from the "response" section of the source document context, while maintaining clarity and conciseness. 
If the exact answer is not found in the context, try to provide a general and informative response related to the question based on the given context.
Avoid making up facts or introducing information not supported by the context.
   CONTEXT: {context}
   
   QUESTION: {question}"""
   
   
   PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
   
   chain_type_kwargs = {"prompt": PROMPT}
   

   chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs
                                   )
   return chain

if __name__=="__main__":
    chain=get_chain()
    print(chain("How to diagnose Parkinson's Disease ?"))