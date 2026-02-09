# Knowledge Conflicts in RAG

LLM의 내부 지식(Parametric Knowledge)과 외부 검색 문서(Retrieved Knowledge) 간의 충돌 문제를 연구하고, ASTUTE RAG 방법론을 구현한 프로젝트입니다.

## 개요

RAG(Retrieval-Augmented Generation) 시스템에서 검색된 외부 문서가 모델의 내부 지식과 충돌할 때, 모델이 어떻게 대응해야 하는지를 연구합니다. ASTUTE RAG는 내부 지식과 외부 지식을 통합하여 일관성 있는 답변을 생성하는 방법론입니다.

## 프로젝트 구조

```
Knowledge-Conflicts/
├── model.py              # LLM 로드 및 추론 (Mistral, Gemini)
├── api_rag.py            # Web 검색 기반 RAG (Serper API)
├── astute_rag.py         # ASTUTE RAG 구현
├── acc_prec.py           # 정확도/정밀도 평가 함수
├── prompt_template.py    # 프롬프트 템플릿 정의
├── retrieval.py          # 문서 검색 관련 함수
├── set_data.py           # 데이터 전처리
├── datasets/             # QA 데이터셋
│   ├── popqa_dataset/
│   ├── nq_dataset/
│   ├── triviaqa_dataset/
│   ├── bioasq_dataset/
│   └── total_qa_sampled/
├── RAG/                  # Wikipedia RAG 서버
│   └── wiki_rag/
├── output/               # 실험 결과
└── results/              # 평가 결과
```

## 지원 모델

- **Mistral-Nemo-Instruct-2407**: 로컬 실행 (4-bit 양자화)
- **Gemini**: Google API를 통한 클라우드 실행

## 데이터셋

4개의 QA 벤치마크 데이터셋을 사용합니다:

| 데이터셋 | 설명 |
|---------|------|
| PopQA | Long tail QA |
| Natural Questions (NQ) | Google 검색 기반 QA |
| TriviaQA | 트리비아 질문 |
| BioASQ | 생의학 QA |



## 참고 문헌

- ASTUTE RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models
