RETRIEVAL_QA_CHAT_TEMPLATE = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know.
Don't make up an answer.
Use three sentences maximum and keep the answer concise.
Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Answer:
""" 