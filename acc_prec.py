import json

#결과 로드
def load_results(filepath):
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results

#정확도 계산 (ground_truth 중 하나라도 answer에 포함되면 정답)
def calculate_accuracy(results):
    correct = 0
    for item in results:
        answer = item["answers"].lower()
        gold_answers = item["ground_truth"]
        if any(str(gold).lower() in answer for gold in gold_answers):
            correct += 1
    return correct / len(results) * 100 if results else 0

#데이터셋별 정확도 계산 dataset_sizes: {"popqa": 260, "nq": 260, "triviaqa": 261, "bioasq": 261}
def calculate_accuracy_by_dataset(results, dataset_sizes):
    accuracies = {}
    start = 0
    for name, size in dataset_sizes.items():
        subset = results[start:start+size]
        accuracies[name] = calculate_accuracy(subset)
        start += size
    accuracies["overall"] = calculate_accuracy(results)
    return accuracies


#검색 precision(정밀도) 계산 함수
def retrieval_precision(passages, gold_answers):
  
    if not passages:
        return 0.0
    
    count = sum(
        any(str(gold).lower() in p.lower() for gold in gold_answers)
        for p in passages
    )
    return count / len(passages)

# 사용 예시 (RAG 결과에 source_documents가 있을 때)
def calculate_precision_by_datasets(results, dataset_sizes):
    precision_dict = {}
    size = 0

    #개별 데이터셋 precision 계산
    for key, value in dataset_sizes.items():
        precision = 0.0
        results_piece = results[size:size+value]
        size += value   
        for result in results_piece:
            passages = [doc['page_content'] for doc in result["docs"]]
            gold_answers = result["ground_truth"]
            precision += retrieval_precision(passages, gold_answers)
        precision = precision / len(results_piece) * 100
        precision_dict[key] = precision

    # 전체 precision 계산
    precision = 0.0
    for result in results:
        passages = [doc['page_content'] for doc in result["docs"]]
        gold_answers = result["ground_truth"]
        precision += retrieval_precision(passages, gold_answers)
    precision = precision / len(results) * 100
    precision_dict["overall"] = precision
    
    return precision_dict