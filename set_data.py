from datasets import load_dataset
import json
import os

# 데이터셋 로드
ds = load_dataset("akariasai/PopQA")

# popqa_dataset 폴더 생성
os.makedirs("popqa_dataset", exist_ok=True)

# 각 split을 JSON 파일로 저장

test_data = ds['test']

# 데이터를 리스트로 변환
data_list = [item for item in test_data]

final_list = []
for idx, data in enumerate(data_list):
    temp_json = {
        "ids" : str(idx),
        "question": data["question"],
        "answers": data["possible_answers"]
    }
    final_list.append(temp_json)


with open("popqa_dataset/qa_dataset.json", 'w', encoding='utf-8') as f: 
    json.dump(final_list, f, ensure_ascii=False, indent=2)