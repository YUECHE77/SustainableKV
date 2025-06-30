import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
# CUDAVISIBLE DEVICES
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from snapkv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral
from SustainableKV.monkeypatch import sustainablekv_replace_mistral, sustainablekv_replace_llama, sustainablekv_replace_mixtral

from fastchat.model import load_model, get_conversation_template

method = 'SustainableKV'  # SustainableKV or SnapKV or None
assert method in ('SustainableKV', 'SnapKV', 'None'), (
    f'Currently we only support SustainableKV or SnapKV, but you are trying to use {method}.'
)

# notification: Llama-2-7b-chat-hf cannot process texts exceed 3500 tokens. Try to use Llama-2-7B-32K-Instruct
model_to_use = 'mistral-7B-instruct-v0.2'
assert model_to_use in (
    'mistral-7B-instruct-v0.2',
    'llama-2-7B-32k-instruct',
    'longchat-v1.5-7b-32k',
    'vicuna-v1.5-7b-16k',
    'lwm-text-chat-1m'
), (
    f'You are using model: {model_to_use} that we do not support.'
)

model2path = json.load(open("/root/SnapKV/experiments/LongBench/config/model2path.json", "r"))
path = model2path[model_to_use]

if method == 'SnapKV':
    replace_llama()
    replace_mistral()
    replace_mixtral()
elif method == 'SustainableKV':
    sustainablekv_replace_llama()
    sustainablekv_replace_mistral()
    sustainablekv_replace_mixtral()
else:
    print('\nYou are using the vanilla model!\n')

question = "\n What is the repository of SnapKV?"
with open('/root/SnapKV/notebooks/snapkv.txt', 'r') as f:
    content = f.read().strip()
# question = 'Who are you?'
# content = ''
query = content + question

model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
    use_cache=True,
    use_flash_attention_2=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    path,
    use_fast=False,
)

if model_to_use == 'mistral-7B-instruct-v0.2':
    conv = get_conversation_template("mistral")
    conv.messages = []
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

elif model_to_use == 'llama-2-7B-32k-instruct' or model_to_use == 'lwm-text-chat-1m':
    prompt = f"[INST] {query} [/INST]"

elif model_to_use == 'vicuna-v1.5-7b-16k' or model_to_use == 'longchat-v1.5-7b-32k':
    conv = get_conversation_template("vicuna")
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

# input_ids = tokenizer.encode(prompt, return_tensors='pt')
# input_ids_len = input_ids.size(1)
input = tokenizer(prompt, truncation=False, return_tensors="pt").to(model.device)
input_ids_len = input.input_ids.shape[-1]

start_time = time.time()
# outputs = model.generate(input_ids.cuda(), max_new_tokens=200, do_sample=False)
outputs = model.generate(
    **input,
    max_new_tokens=200,
    num_beams=1,
    do_sample=False,
    temperature=1.0,
    min_length=input_ids_len+1,
)
end_time = time.time()

ans = tokenizer.decode(outputs[0][input_ids_len:], skip_special_tokens=True).strip()

print("\n------------- Model's answer:-------------\n")
print(f'USER: {question.strip()}')
print(f'ASSISTANT: {ans}')
print('\n------------------------------------------')

print(f'Total time: {end_time - start_time}\n')
