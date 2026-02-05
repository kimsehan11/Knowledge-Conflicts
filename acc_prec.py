import json

#결과 로드
#output/output_with_base_rag.jsonl <- 이 결과 불러오는거임
def load_results(filepath):
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results

#accuracy 계산 
def calculate_accuracy(results:list):
    correct = 0
    for item in results:
        answer = item["answers"].lower()
        gold_answers = item["ground_truth"]
        if any(str(gold).lower() in answer for gold in gold_answers):
            correct += 1
    return correct / len(results) * 100 if results else 0

#데이터셋별 accuracy 계산 
#dataset_sizes: {"popqa": 260, "nq": 260, "triviaqa": 261, "bioasq": 261}
def calculate_accuracy_by_dataset(results, dataset_sizes):
    accuracies = {}
    start = 0
    for name, size in dataset_sizes.items():
        subset = results[start:start+size]
        accuracies[name] = calculate_accuracy(subset)
        start += size
    accuracies["overall"] = calculate_accuracy(results)
    return accuracies


#precision 계산 함수
def retrieval_precision(passages, gold_answers):
  
    if not passages:
        return 0.0
    
    count = sum(
        any(str(gold).lower() in p.lower() for gold in gold_answers)
        for p in passages
    )
    return count / len(passages)

# 데이터셋별 precision 계산
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

#austute rag 정확도 계산
def calculate_accuracy_by_dataset_with_astute_rag(results,dataset_sizes,answers):
    cur_line = 0 

    acc_dict = {"popqa": 0, "nq": 0, "triviaqa": 0, "bioasq": 0}
    
    for key,value in dataset_sizes.items():
        next_line = cur_line + value
        for idx in range(cur_line, next_line):
            result = results[idx]
            
            # 모델 응답 전체에서 gold answer 포함 여부 확인
            if any(str(res).lower() in answers[idx].lower() for res in result['ground_truth']):
                acc_dict[key] += 1
        cur_line = next_line
    
       
    for key in acc_dict.keys():
        acc_dict[key] = round(acc_dict[key] / dataset_sizes[key] * 100 , 1) 
        
    return acc_dict

#각 테스트 케이스의 Retrieval Precision 계산
def evaluate_retrieval_prec_for_questions(data_path):
    with open(data_path,'r') as f:
        precisions = []
        #우선 한 줄마다의 사전을 받고
        for line in f:
            test_dict = json.loads(line.strip())

            #만약 docs가 비어있다면 패스
            if len(test_dict.get("docs")) == 0:
                precisions.append(0) 
                continue
            
            #한 줄 사전에서 docs_list 부분을 따로 뽑아내고, ground_truth도 뽑아내서 ,
            count = 0
            docs_list = [page_content_dict["page_content"].lower() for page_content_dict in test_dict.get("docs")]
            ground_truth = test_dict.get("ground_truth")

            #ground_truth에 있는 정답이 docs_list에 포함되어 있는 개수를 세면 됨.
            for doc in docs_list:
                if any(str(gold_answer).lower() in doc for gold_answer in ground_truth):
                    count += 1
            
            precisions.append(count / len(docs_list))
    
    return precisions

#precision 별로 테스트 케이스 나누기
def split_precision_group(output_path,prec_list):

    with open(output_path, "r") as f:

        f = f.readlines()

        precision_test_dict = {'<= 0.1' : [],'<= 0.2' : [] , '<= 0.3' : [] ,'<= 0.5' : [], '<= 0.6' : [] ,'<= 0.7' : [] ,'<= 0.8' : [] , '<= 0.9' : [],'<= 1.0' : [] }

        for idx,prec in enumerate(prec_list):
            if prec <= 0.1:
                precision_test_dict['<= 0.1'].append(json.loads(f[idx]))
            elif prec <= 0.2:
                precision_test_dict['<= 0.2'].append(json.loads(f[idx]))
            elif prec <= 0.3:
                precision_test_dict['<= 0.3'].append(json.loads(f[idx]))
            elif prec <= 0.5:
                precision_test_dict['<= 0.5'].append(json.loads(f[idx]))
            elif prec <= 0.6:
                precision_test_dict['<= 0.6'].append(json.loads(f[idx]))
            elif prec <=0.7:
                precision_test_dict['<= 0.7'].append(json.loads(f[idx]))
            elif prec <= 0.8:
                precision_test_dict['<= 0.8'].append(json.loads(f[idx]))
            elif prec <= 0.9:
                precision_test_dict['<= 0.9'].append(json.loads(f[idx]))
            else:
                precision_test_dict['<= 1.0'].append(json.loads(f[idx]))

    return precision_test_dict


            

