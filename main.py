import os
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")


def load_documents():
    file_path = os.path.join(os.getcwd(), "mediumblog2.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents

def split_documents(docs: List[Document]):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_documents(chunks: List[Document]):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)
    return vectorstore

if __name__ == "__main__":
    print("Hello from embeddings-and-vector-db!")
    docs = load_documents()
    chunks = split_documents(docs)
    vectorstore = embed_documents(chunks)
