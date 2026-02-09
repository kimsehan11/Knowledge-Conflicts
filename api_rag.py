import http.client
import json
from dotenv import load_dotenv
import os
import requests
from bs4 import BeautifulSoup
from prompt_template import RAG_PROMPT_TEMPLATE
from langchain_core.prompts import PromptTemplate 
from model import llm_answer, llm_answer_gemini 

load_dotenv()

def search_serper(query, pages=3):
    all_results = []
    for page in range(1, pages + 1):
        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({"q": query, "page": page})
        headers = {
            'X-API-KEY': os.getenv("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        all_results.extend(data.get("organic", []))
    return all_results

def is_accessible(url, timeout=10):
    try:
        r = requests.get(url, timeout=timeout, allow_redirects=True,
                        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        return r.status_code == 200, r.text
    except:
        return False, None

def extract_paragraph(html, snippet):
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    sentences = text.replace('\n', ' ').split('.')
    
    keywords = snippet.replace('...', '').strip().split()[:5]
    keyword_str = ' '.join(keywords)
    
    for sentence in sentences:
        if keyword_str in sentence:
            return sentence.strip() + '.'
    return snippet

def get_accessible_results(query, target=10, pages=3):
    results = search_serper(query, pages)
    accessible = []
    for r in results:
        if len(accessible) >= target:
            break
        if "snippet" not in r:
            continue
        success, html = is_accessible(r["link"])
        if success:
            r["paragraph"] = extract_paragraph(html, r["snippet"])
            accessible.append(r)
    return accessible

def web_rag(query, llm, target=10, mode="mistral"):
    RAG_PROMPT = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    docs = get_accessible_results(query, target=target, pages=3)
    context = "\n\n".join([f"[{i+1}] {doc['title']}\n{doc['paragraph']}" for i, doc in enumerate(docs)])
    prompt = RAG_PROMPT.format(context=context, question=query)
    if mode == "mistral":
        response = llm_answer(llm[0], llm[1], prompt)
    elif mode == "gemini":
        response = llm_answer_gemini(llm, prompt)
    
    return {
        "query": query,
        "answer": response,
        "source_documents": docs
    }

