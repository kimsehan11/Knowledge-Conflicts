# Knowledge-Conflicts
# Averitec 데이터셋 벡터 DB 변환 가이드

## 로컬 PC에서 실행하기

### 1. 파일 다운로드

서버에서 로컬 PC로 스크립트 복사:
```bash
# 로컬 PC에서 실행
scp sehan@서버주소:/home/sehan/sehan_workspace/averitec_to_vectordb.py ~/Downloads/
scp sehan@서버주소:/home/sehan/sehan_workspace/requirements_vectordb.txt ~/Downloads/
```

### 2. 환경 설정

```bash
cd ~/Downloads

# 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 라이브러리 설치
pip install -r requirements_vectordb.txt
```

### 3. 스크립트 수정

`averitec_to_vectordb.py` 파일을 열고 경로 수정:

```python
# 메인 함수에서 데이터 경로 수정
DATA_DIR = "/home/humane/Downloads/dev_knowledge_store"  # 실제 경로로 수정
DB_PATH = "./averitec_chroma_db"  # 벡터 DB 저장 경로
```

### 4. 실행

```bash
python averitec_to_vectordb.py
```

## 서버에서 실행하기

데이터를 서버로 업로드 후 실행:

```bash
# 로컬 PC에서 데이터 업로드
scp -r /home/humane/Downloads/dev_knowledge_store sehan@서버주소:/home/sehan/sehan_workspace/

# 서버에서 실행
ssh sehan@서버주소
cd /home/sehan/sehan_workspace
pip install -r requirements_vectordb.txt
python averitec_to_vectordb.py
```

## 벡터 DB 사용하기

벡터 DB 생성 후 검색 예제:

```python
from averitec_to_vectordb import AveritecVectorDB

# 기존 벡터 DB 로드
vectordb = AveritecVectorDB(
    embedding_type="sentence_transformers",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    db_path="./averitec_chroma_db"
)

# 기존 컬렉션 로드
vectordb.collection = vectordb.client.get_collection("averitec_collection")

# 검색
results = vectordb.search("your query here", n_results=5)

for i, doc in enumerate(results['documents'][0]):
    print(f"{i+1}. {doc[:200]}...")
```

## 임베딩 모델 선택

### Sentence Transformers (무료, 추천)
- **다국어**: `paraphrase-multilingual-MiniLM-L12-v2`
- **영어**: `all-MiniLM-L6-v2` (가볍고 빠름)
- **영어 고성능**: `all-mpnet-base-v2`

### OpenAI (유료)
- `text-embedding-3-small`: 저렴하고 빠름
- `text-embedding-3-large`: 고성능

## 데이터 구조 커스터마이징

Averitec 데이터셋의 실제 구조에 맞게 `process_document` 메서드를 수정하세요:

```python
def process_document(self, doc: Dict) -> Dict:
    data = doc.get('data', {})

    # 실제 필드명에 맞게 수정
    text = f"{data.get('claim', '')} {data.get('evidence', '')}"

    metadata = {
        'claim_id': data.get('claim_id'),
        'label': data.get('label'),
        # 필요한 메타데이터 추가
    }

    return {'text': text, 'metadata': metadata}
```