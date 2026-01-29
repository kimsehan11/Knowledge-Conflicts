import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#데이터 로드
def data_load():
    with open("total_qa_sampled/qa_dataset.json", "r") as f:
        qa_dataset = json.load(f)
    return qa_dataset

#모델 로드
def llm_load():
    model_id = "./Mistral-Nemo-Instruct-2407"
    tokenizer = AutoTokenizer.from_pretrained(model_id, fix_mistral_regex=True)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.float16
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config).to(device) 

    return model, tokenizer

#모델 답변
def llm_answer(model, tokenizer, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,  
        do_sample=True,     
        pad_token_id=tokenizer.eos_token_id,
        temperature=0
    )

    answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # 메모리 정리
    del inputs, outputs
    torch.cuda.empty_cache()

    return answer

#모델 답변 리스트 생성
def llm_generate_list(model, tokenizer, qa_dataset):
    internal_pessages = []
    for idx,qa in enumerate(qa_dataset):
        prompt = f"""
        Generate a document that provides accurate and relevant information to answer the given question.
        If the information is unclear or uncertain, explicitly state ’I don’t know’ to avoid any hallucinations.
        Question: {qa["question"]} Document
        """
        internal_pessage = { "ids" : qa["ids"], "question": qa["question"] , "model_answer": llm_answer(model, tokenizer, prompt)}
        internal_pessages.append(internal_pessage)
        if idx < 10:
            print(internal_pessage)
            print()
    return internal_pessages

#모델 병합 (internal_pessages 와 external_pessages 병합) 아직 미완성 
def combine_pessages(internal_pessages,external_pessages):
    combined_pessages = []
    for internal_pessage, external_pessage in zip(internal_pessages, external_pessages):
        combined_pessage = {
            "ids": internal_pessage["ids"],
            "question": internal_pessage["question"],
            "combined_answer" : external_pessage[""] + " " + internal_pessage["answer"]
        }
        combined_pessages.append(combined_pessage)
    
    print(combined_pessages[:5] + "\n")
    return combined_pessages