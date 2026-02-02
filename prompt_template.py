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

P_GEN = """Generate a document that provides accurate and relevant information to answer the given question.
If the information is unclear or uncertain, explicitly state ’I don’t know’ to avoid any hallucinations.
Question: {question} Document:"""

P_CON = """Task: Consolidate information from both your own memorized documents and externally retrieved
documents in response to the given question.
* For documents that provide consistent information, cluster them together and summa-
rize the key details into a single, concise document.
* For documents with conflicting information, separate them into distinct documents, ensuring
each captures the unique perspective or data.
* Exclude any information irrelevant to the query.
For each new document created, clearly indicate:
* Whether the source was from memory or an external retrieval.
* The original document numbers for transparency.
Initial Context: {context_init}
Last Context: {context}
Question: {question}
New Context:
"""

P_ANS="""Task: Answer a given question using the consolidated information from both your own memorized
documents and externally retrieved documents.
Step 1: Consolidate information
* For documents that provide consistent information, cluster them together and summarize the key
details into a single, concise document.
* For documents with conflicting information, separate them into distinct documents, ensuring
each captures the unique perspective or data.
* Exclude any information irrelevant to the query.
For each new document created, clearly indicate:
* Whether the source was from memory or an external retrieval.
* The original document numbers for transparency.
Step 2: Propose Answers and Assign Confidence
For each group of documents, propose a possible answer and assign a confidence score based on
the credibility and agreement of the information.
Step 3: Select the Final Answer
After evaluating all groups, select the most accurate and well-supported answer.
Highlight your exact answer within <ANSWER> your answer </ANSWER>.

Initial Context: {context_init}
[Consolidated Context: {context}] # optional
Question: {question}
Answer:

"""