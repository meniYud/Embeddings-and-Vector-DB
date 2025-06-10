import os
import requests
from typing import List
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from langchain_community.document_loaders.firecrawl import FireCrawlLoader

pinecone_api_key = os.getenv("PINECONE_API_KEY")
# index_name = os.getenv("PINECONE_INDEX_URL")
openai_api_key = os.getenv("OPENAI_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

def load_documents_fc():
    url = "https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/"
    
    # Use FireCrawl API directly
    headers = {
        'Authorization': f'Bearer {firecrawl_api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'url': url,
        'formats': ['markdown', 'html']  # Adjust based on what you need
    }
    
    response = requests.post(
        'https://api.firecrawl.dev/v1/scrape',
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        data = response.json()
        content = data.get('data', {}).get('markdown', '') or data.get('data', {}).get('content', '')
        
        # Create Document object compatible with LangChain
        document = Document(
            page_content=content,
            metadata={
                'source': url,
                'title': data.get('data', {}).get('title', ''),
                'description': data.get('data', {}).get('description', '')
            }
        )
        return [document]
    else:
        raise Exception(f"FireCrawl API error: {response.status_code} - {response.text}")

def load_documents():
    # file_path = os.path.join(os.getcwd(), "mediumblog2.txt")
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return documents

def split_documents(docs: List[Document]):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks

def embed_documents(chunks: List[Document], index_name: str):
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vectorstore = PineconeVectorStore.from_documents(documents=chunks, embedding=embeddings, index_name=index_name)
    return vectorstore

def main(useFC: bool = False):
    print("Hello from ingestion.py")
    if not useFC:
        index_name = os.getenv("PINECONE_INDEX_NAME")
        docs = load_documents()
        chunks = split_documents(docs)
        vectorstore = embed_documents(chunks, index_name)
    else:
        index_name = os.getenv("PINECONE_AE_INDEX_NAME")
        docs = load_documents_fc()
        # chunks = split_documents(docs)
        # vectorstore = embed_documents(chunks, index_name)

    

if __name__ == "__main__":
    main(useFC=True)