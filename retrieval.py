import os
from typing import List, Tuple
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
openai_api_key = os.getenv("OPENAI_API_KEY")



def main(query: str, chat_history: List[dict[str, str]]):
    print(f"started retrieval for the query: {query} with {'empty' if not chat_history else 'non-empty'} chat history")
    embedding = OpenAIEmbeddings(api_key=openai_api_key)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embedding,
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    rephrased_query = hub.pull("langchain-ai/chat-langchain-rephrase")
    history_aware_retriever = create_history_aware_retriever(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=rephrased_query,
    )

    retrieval_qa_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain,
    )

    result = retrieval_qa_chain.invoke(input={"input": query, "chat_history": chat_history})

    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source_documents": result["context"],
    }
    return new_result
    
    # A: (result["answer"])
    # AlphaEvolve is an AI system introduced by Google DeepMind in 2025 that demonstrates asynchronous intelligence,
    # characterized by the effective coordination of cognitive processes operating at different temporal speeds.
    # It achieved significant breakthroughs, including improvements to matrix multiplication algorithms,
    # discoveries in mathematical constructions, and optimization of critical infrastructure systems.
    # Instead of relying on revolutionary new technologies,
    # AlphaEvolve's intelligence emerges from the collaboration of various processes functioning across different time scales,
    # showcasing a new model for human-AI collaboration.
    
    
    
if __name__ == "__main__":
    main(query="What is AlphaEvolve?")


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