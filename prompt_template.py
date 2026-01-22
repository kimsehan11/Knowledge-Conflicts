# RAG 프롬프트 템플릿
RAG_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

#No RAG 프롬프트 템플릿
NO_RAG_PROMPT_TEMPLATE = """Generate a document that provides accurate and relevant information to answer the given question.
If the information is unclear or uncertain, explicitly state ’I don’t know’ to avoid any hallucinations.
Question: {question}
Answer:"""