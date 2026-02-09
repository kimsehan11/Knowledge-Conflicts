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

## 설치

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (.env 파일 생성)
SERPER_API_KEY=your_serper_api_key
GOOGLE_API_KEY=your_google_api_key  # Gemini 사용 시
```

## 주요 기능

### 1. 기본 RAG
```python
from api_rag import web_rag
from model import llm_load

llm = llm_load()
result = web_rag("What is the capital of France?", llm)
```

### 2. ASTUTE RAG
내부 지식 생성 → 외부 문서 검색 → 지식 통합 → 답변 생성

```python
from astute_rag import make_internal_passage, combine_passage, finalize_answer
```

### 3. 평가
```python
from acc_prec import calculate_accuracy_by_dataset, calculate_precision_by_datasets

# 정확도 계산
dataset_sizes = {"popqa": 260, "nq": 260, "triviaqa": 261, "bioasq": 261}
accuracies = calculate_accuracy_by_dataset(results, dataset_sizes)
```

## 평가 지표

- **Accuracy**: 모델 응답이 정답을 포함하는 비율
- **Retrieval Precision**: 검색된 문서 중 정답을 포함하는 문서의 비율

## 참고 문헌

- ASTUTE RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models
