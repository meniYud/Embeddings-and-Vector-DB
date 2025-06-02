import os
from typing import List
from dotenv import load_dotenv
from templates import RETRIEVAL_QA_CHAT_TEMPLATE

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")

def format_documents(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def main():
    print("Hello from retrieval.py")
    embedding = OpenAIEmbeddings(api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

    query = "What is AlphaEvolve?"
    
    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding,
    )

    custom_rag_prompt = PromptTemplate.from_template(
        template=RETRIEVAL_QA_CHAT_TEMPLATE
    )

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_documents, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )


    result = rag_chain.invoke(query)
    print(result)
    
    # A:
    # AlphaEvolve is an AI system introduced by Google DeepMind in 2025 that achieved significant advancements in various fields,
    # including mathematics and infrastructure optimization, by employing asynchronous intelligence.
    # It operates through the coordination of cognitive processes that function at different temporal scales,
    # demonstrating a new model for human-AI collaboration.
    # This approach emphasizes the elegance of combining different cognitive specialties rather than relying solely on
    # powerful individual components.
    # Thanks for asking!
    
    
    
if __name__ == "__main__":
    main()


# without augmentation:
# Q:
# query = "What is AlphaEvolve?"
# llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
# chain = PromptTemplate.from_template(template=query).pipe(llm)
# result = chain.invoke(input={"query": query})
# print(result.content)

# A:
# As of my last knowledge update in October 2023, there isn't a widely recognized entity or concept specifically known as "AlphaEvolve."
# It's possible that it could refer to a company, service, product, or a term that has emerged after my last update.

# If "AlphaEvolve" is indeed a recent development or a niche term, I would recommend checking the latest sources,
# such as news articles, company websites, or social media platforms for the most current information. If you have any
# specific context or details regarding what "AlphaEvolve" pertains to, please share,
# and I might be able to provide more relevant information or insights!
