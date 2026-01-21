from datasets import load_dataset
import random
import json
import os

#popqa 데이터셋 로드
def load_popqa(split="test", output_dir="datasets/popqa_dataset"):
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

    with open(f"{output_dir}/qa_dataset_origin.json", 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    return final_list

#popqa 데이터셋 전처리 함수
def preprocess_popqa_answers(test_data, output_dir="datasets/popqa_dataset"):
    for item in test_data:
        item["answers"] = json.loads(item["answers"])

    if not os.path.exists(f"{output_dir}/qa_dataset.json"):
        with open(f"{output_dir}/qa_dataset.json", 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
            
    return test_data

#nq 데이터셋 로드
def load_nq(split="validation", output_dir="datasets/nq_dataset"):
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
def preprocess_nq_answers(test_data, output_dir = "datasets/nq_dataset"):
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
def load_triviaqa(split="validation", output_dir="datasets/triviaqa_dataset"):
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

#bioasq 데이터셋 전처리
def preprocess_bioasq_answers(test_data, output_dir="datasets/bioasq_dataset"):
    all_answers = []
    test_list = [item for item in test_data["questions"] if "exact_answer" in item and item["type"] != "yesno"]

    for idx, item in enumerate(test_list):
        temp = []
        question = item["body"]
        answer = item["exact_answer"]
        for ans in answer:
            if isinstance(ans, list):
                temp.extend(ans)
            else:
                temp.append(ans)
        answer = temp
        all_answers.append({
            "ids": str(idx),
            "question": question,
            "answers": answer
        })

    if not os.path.exists(os.path.join(output_dir,"qa_dataset.json")):
        with open(os.path.join(output_dir, "qa_dataset.json"), "w", encoding="utf-8") as f:
            json.dump(all_answers, f, ensure_ascii=False, indent=2)

    return all_answers

    

#무작위로 num_qa개 샘플링
def extract_random_qa(test_data, num_qa=260):
    samples = random.sample(test_data, num_qa)
    return sorted(samples, key=lambda x: int(x['ids']))
