from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from model import llm_answer, llm_answer_batch
from prompt_template import P_ANS
import re
import json


def make_prompt_template(template, input_vars):
    return PromptTemplate(template=template, input_variables=input_vars)

#external passage 리스트 생성 함수
def make_external_passage(filepath="output/output_with_base_api_rag_2.jsonl"):
    E = []

    with open(filepath,"r") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            E.append(data["docs"])

    return E

#internal passage 리스트 생성 함수
def make_internal_passage(q, P_gen, M):
    I = []

    for each_q in tqdm(q[:], desc = "making Internal passage" ):
        P_gen_prompt = P_gen.format(question=each_q["question"])
        I.append(llm_answer(M[0],M[1],P_gen_prompt))

    return I

#internal passage 리스트 생성 함수 (배치 버전)
def make_internal_passage_batch(q, P_gen, M, batch_size=4):
    # 모든 프롬프트 미리 생성
    prompts = [P_gen.format(question=each_q["question"]) for each_q in q]
    
    # 배치로 처리
    I = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="making Internal passage (batch)"):
        batch_prompts = prompts[i:i+batch_size]
        batch_answers = llm_answer_batch(M[0], M[1], batch_prompts, batch_size=batch_size)
        I.extend(batch_answers)
    
    return I

#최종 답변 생성 함수 (배치 버전)
def finalize_answer_batch(llm, questions, contexts, batch_size=4, P_ans=None):
    if P_ans is None:
        P_ans = make_prompt_template(P_ANS, ["question", "context_init", "context"])
    
    # 모든 프롬프트 미리 생성
    prompts = [
        P_ans.format(question=q, context_init=ctx, context=None) 
        for q, ctx in zip(questions, contexts)
    ]
    
    # 배치로 처리
    answers = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="generating final answers (batch)"):
        batch_prompts = prompts[i:i+batch_size]
        batch_answers = llm_answer_batch(llm[0], llm[1], batch_prompts, batch_size=batch_size)
        answers.extend(batch_answers)
    
    return answers

#passage 병합하는 함수
def combine_passage(E, I):
    combine_passages = []

    for idx, e in enumerate(E):
        I[idx] = re.sub(r'\"', '', I[idx]).strip()
        # internal passage도 external과 동일한 형식으로 통일
        internal_doc = {'page_content': I[idx], 'source': 'internal'}
        # external docs에 source 표시 추가
        external_docs = [{'page_content': doc.get('page_content', ''), 'source': 'external'} for doc in e]
        combined_passage = external_docs + [internal_doc]
        combine_passages.append(combined_passage)

    return combine_passages

#프롬프트 생성 함수
def make_prompts(P_GEN, P_CON, P_ANS):
    P_gen = PromptTemplate(template=P_GEN, input_variables=["question"])
    P_con = PromptTemplate(template=P_CON, input_variables=["question", "retrieved_passages"])
    P_ans = PromptTemplate(template=P_ANS, input_variables=["question", "context_init","context"])
    return P_gen, P_con, P_ans

#페이지 소스 생성 함수
def make_passage_source(combine_passages):
    passage_sources = []
    for p in combine_passages:
        passage_source = [1 if doc.get('source') == 'internal' else 0 for doc in p]
        passage_sources.append(passage_source)

    return passage_sources

#최종 답변 생성 함수
def finalize_answer(llm, question, context_init, context=None,P_ans=make_prompt_template(P_ANS,["question", "context_init","context"])):
    P_ans_prompt = P_ans.format(
        question=question,
        context_init=context_init,
        context=context
    )
    answer = llm_answer(llm[0], llm[1], P_ans_prompt)
    return answer

