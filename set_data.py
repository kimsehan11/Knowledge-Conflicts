from datasets import load_dataset
import random
import json
import os

#popqa 데이터셋 로드
def load_popqa(split="test", output_dir="popqa_dataset"):
    ds = load_dataset("akariasai/PopQA", split=split)
    os.makedirs(output_dir, exist_ok=True)

    final_list = []
    for idx, data in enumerate(ds):
        temp_json = {
            "ids": str(idx),
            "question": data["question"],
            "answers": data["possible_answers"]
        }
        final_list.append(temp_json)

    with open(f"{output_dir}/qa_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    return final_list

#nq 데이터셋 로드
def load_nq(split="validation", output_dir="nq_dataset"):
    ds = load_dataset("google-research-datasets/natural_questions", "default", split=split)
    os.makedirs(output_dir, exist_ok=True)

    final_list = []
    for idx, data in enumerate(ds):
        answers = [ans['text'] for ans in data['annotations']['short_answers'] if ans['text']]
        if not answers:
            continue
        temp_json = {
            "ids": str(idx),
            "question": data["question"]["text"],
            "answers": answers
        }
        final_list.append(temp_json)

    with open(f"{output_dir}/qa_dataset_origin.json", 'w', encoding='utf-8') as f:
        final_list = preprocess_nq_answers(final_list)
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    return final_list

#nq_dataset 전처리 -> popqa와 동일한 형식으로 맞추기
def preprocess_nq_answers(test_data, output_dir = "nq_dataset"):
    for item in test_data:
        if len(item["answers"]) > 1:
            temp = []
            for ans in item["answers"]:
                temp += ans
                temp = list(set(temp))
            item["answers"] = temp
    
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(f"{output_dir}/qa_dataset.json"):
        with open(f"{output_dir}/qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)

    return test_data


# TriviaQA 데이터셋 로드
def load_triviaqa(split="validation", output_dir="triviaqa_dataset"):
    ds = load_dataset("trivia_qa", "rc", split=split)
    os.makedirs(output_dir, exist_ok=True)

    final_list = []
    for idx, data in enumerate(ds):
        temp_json = {
            "ids": str(idx),
            "question": data["question"],
            "answers": data["answer"]["aliases"]
        }
        final_list.append(temp_json)

    with open(f"{output_dir}/qa_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    return final_list

#무작위로 num_qa개 샘플링
def extract_random_qa(test_data, num_qa=260):
    samples = random.sample(test_data, num_qa)
    return sorted(samples, key=lambda x: int(x['ids']))
