import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

def main():
    print("Hello from chatty_pdf.py")
    pdf_path = os.path.join(os.path.dirname(__file__), "assets", "AI Agents-Top 25 Use Cases Transforming Industries.pdf")
    loader = PyPDFLoader(file_path=pdf_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    chunks = text_splitter.split_documents(documents=docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vectorstore.save_local("agents_25_use_cases")
    
if __name__ == "__main__":
    main()
    