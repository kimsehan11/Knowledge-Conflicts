from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings 
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from prompt_template import RAG_PROMPT_TEMPLATE
from model import llm_load, llm_answer, llm_generate_list


#벡터스토어 로드
def load_vectorstore(vectordb_path="./vectorDB"):
    print(f"벡터스토어 로드중")

    embeddings = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cuda"},
        query_instruction="Represent this sentence for retrieving relevant passages:",
    )

    vectorstore = FAISS.load_local(
        vectordb_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore, embeddings

#검색기 로드
def load_retriever(vectordb_path="./vectorDB"):

    vectorstore, embeddings = load_vectorstore(vectordb_path)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})
    return retriever

#쿼리 질문하기 - Score이랑 Debug (검색기 디버깅용 함수)
def test_queries(vectorstore, queries, k=5,score=False,debug=False):

    if not score:
        for i, query in enumerate(queries, 1):
            if debug:
                print(f"Query {i}: {query}")
        
            results = vectorstore.similarity_search(query, k=k)

            if debug:
                for j, doc in enumerate(results, 1):
                    print(f"\n[Result {j}]")
                    print(f"Title: {doc.metadata.get('title', 'N/A')}")
                    print(f"URL: {doc.metadata.get('url', 'N/A')}")
                    print(f"Content preview: {doc.page_content[:300]}...")
                    print(f"-" * 80)            
    else:
        for i, query in enumerate(queries, 1):
            if debug:
                print(f"Query {i}: {query}")
        
            results = vectorstore.similarity_search_with_score(query, k=k)

            if debug:
                for j, (doc, score_val) in enumerate(results, 1):  # 튜플 언패킹
                    print(f"\n[Result {j}] (score: {score_val:.4f})")
                    print(f"Title: {doc.metadata.get('title', 'N/A')}")
                    print(f"URL: {doc.metadata.get('url', 'N/A')}")
                    print(f"Content preview: {doc.page_content[:300]}...")
                    print(f"-" * 80)

#Retrival Augmented Generation 구현
def rag(vectorstore, query, llm, k=5):

    RAG_PROMPT = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([ f"[{i+1}] {doc.metadata.get('title', 'N/A')}\n{doc.page_content}"  for i, doc in enumerate(docs)])
    prompt = RAG_PROMPT.format(context=context, question=query)
    response = llm_answer(llm[0], llm[1], prompt)
    
    return {
        "query": query,
        "answer": response,
        "source_documents": docs
    }



