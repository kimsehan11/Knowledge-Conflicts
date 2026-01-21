from datasets import load_dataset
import json
import os


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

    with open(f"{output_dir}/qa_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)
    return final_list


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
