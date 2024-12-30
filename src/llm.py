from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import transformers
import torch
from huggingface_hub import login
import sys
import os

def get_paths_from_string(llm_string):

    path_dict = { ## change to wherever your LLM files are located
        "llama-7b": "/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf/",
        "llama-13b": "/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf/",
        "llama-70b": "/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf/",
        "mistral-7b": "/data/locus/project_data/project_data2/dylansam/Mistral-7B-Instruct-v0.2",
        "mistral-8x7b": "/data/models/huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1",
        "llama3-8b": "/data/models/huggingface/meta-llama/Llama-3.1-8B-Instruct/",
        "llama3-3b": "/data/models/huggingface/meta-llama/Llama-3.2-3B-Instruct/",
        "llama3-70b": "/data/models/huggingface/meta-llama/Llama-3.1-70B-Instruct/",
    }

    return path_dict[llm_string]

def get_left_pad(llm_string):

    # check if left padding or right padding
    if "llama" in llm_string:
        left_pad = False
    elif "mistral" in llm_string:
        left_pad = True
    else:
        left_pad = False # default to left pad...
    return left_pad

def get_add_token(llm_string):

    if "llama" in llm_string:
        add_token = True
    elif "mistral" in llm_string:
        add_token = True
    return add_token

def load_llm(llm_string):

    path = get_paths_from_string(llm_string)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.float16)
    return model, tokenizer