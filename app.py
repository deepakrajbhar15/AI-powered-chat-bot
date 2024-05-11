from dotenv import load_dotenv
load_dotenv()

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


from langchain.prompts import PromptTemplate


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_pdf_text(pdf_docs):

    text=""
    for pdf in pdf_docs:
        # pdf_stream = io.BytesIO(pdf)
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
            
    return text

# import fitz  # PyMuPDF

# def extract_text_from_pdf(pdf_path):
#     with fitz.open(pdf_path) as pdf_document:
#         text = ""
#         for page_num in range(pdf_document.page_count):
#             page = pdf_document[page_num]
#             text += page.get_text()
#     return text

 



# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         with pdfplumber.open(pdf) as pdf_document:
#             for page in pdf_document.pages:
#                 text += page.extract_text()
#     return text

def get_text_chunks(text):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks=text_splitter.split_text(text)
    return chunks
def get_vector_store(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_stores=FAISS.from_texts(chunks,embedding=embeddings)
    vector_stores.save_local("faiss_index")
    
    
def get_conversational_chain():
    prompt_template="""
    answer the qeustions as detailed as possible from the provided context make sure to provided context is correct if don't know answer just 
    say," answer is not avaliable in the context", don't provide the wromg answer.
    context:\n {context}?\n
    questions:\n {question}\n
    answer:
    
    
    
    """
    model=ChatGoogleGenerativeAI(model='gemini-pro',temperature=0.3)
    prompt=PromptTemplate(template=prompt_template,input_variables=['context','question'])
    
    chain=load_qa_chain(model,chain_type='stuff',prompt=prompt)
    return chain

def user_input(user_question):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)
    docs=new_db.similarity_search (user_question)
    chain=get_conversational_chain()
    
    response=chain(
        
    {
        "input_documents":docs ,"question": user_question
    },return_only_outputs=True
    )
    
    print(response)
    st.write("reply:",response["output_text"])
    
def main():
    st.set_page_config("chat with multple pdf")
    st.header("chat with multiple pdf using geminiai")
    user_question= st.text_input("Ask a question from pdfs" )
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title("menu:")
        pdf_docs=st.file_uploader("upload your pdf files amd click on the submit button & proceed",accept_multiple_files=True)
        if st.button('submit & process'):
            with st.spinner("processing.."):
                
                raw_text=get_pdf_text(pdf_docs)
                text_chunks=get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("done")
                
if __name__=="__main__":
    main()