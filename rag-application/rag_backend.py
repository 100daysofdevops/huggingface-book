import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

def create_policy_index():
    print("Loading the pdf file...")
    pdf_loader=PyPDFLoader("aws_ec2_faq.pdf")

    print("Splitting text from PDF file...")
    text_splitter=RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)

    print("Loading Hugging Face model for embeddings...")
    embeddings_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    print("Creating ChromaDB vector store...")
    vector_store_creator=VectorstoreIndexCreator(
        text_splitter=text_splitter,
        embedding=embeddings_model,
        vectorstore_cls=Chroma
    )

    print("Creating index from loaded pdf...")
    index=vector_store_creator.from_loaders([pdf_loader])

    print("Index created sucessfully")
    return index

def load_llm():
    print("Loading LLM...")
    llm_pipeline=pipeline(
        "text-generation",
        model="EleutherAI/gpt-neo-2.7B",
        device=0 if torch.cuda.is_available() else -1,
        max_new_tokens=50,
        clean_up_tokenization_spaces=True
    )

    llm_model=HuggingFacePipeline(pipeline=llm_pipeline)
    print("LLM model loaded sucessfully...")
    return llm_model    

def retrieve_response(index, query):
    llm=load_llm()
    response=index.query(question=query, llm=llm)
    return response

if __name__=='__main__':
    index=create_policy_index()
    llm=load_llm()
