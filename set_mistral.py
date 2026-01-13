import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import tqdm as notebook_tqdm

model_id = "mistralai/Mistral-Nemo-Base-2407"
tokenizer = AutoTokenizer.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

inputs = tokenizer("Hello my name is", return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))