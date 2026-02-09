import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from google import genai
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

#모델 로드(제미나이)
def llm_load_gemini():
    client = genai.Client()
    return client

#모델 답변(제미나이)
def llm_answer_gemini(client, prompt, model="gemini-2.5-flash-lite"): #gemini-3-flash-preview #gemini-2.5-flash
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text

#모델 답변
def llm_answer(model, tokenizer, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)
    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,  
        do_sample=False,     
        pad_token_id=tokenizer.eos_token_id,
    )

    answer = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    
    # 메모리 정리
    del inputs, outputs
    torch.cuda.empty_cache()

    return answer

#모델 답변 (배치 처리)
def llm_answer_batch(model, tokenizer, prompts, batch_size=4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    answers = []
    
    # pad_token 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # padding_side를 tokenizer 속성으로 설정
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=4096, 
            padding=True,
            padding_side='left'
        ).to(device)
        
        # 각 입력의 실제 길이 저장 (패딩 제외)
        input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1).tolist()
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        for j, output in enumerate(outputs):
            answer = tokenizer.decode(output[input_lengths[j]:], skip_special_tokens=True)
            answers.append(answer)
        
        del inputs, outputs
        torch.cuda.empty_cache()

    # padding_side 원복
    tokenizer.padding_side = original_padding_side
    
    return answers

