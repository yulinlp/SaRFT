import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig, pipeline, set_seed
from peft import PeftModel, PeftConfig
from tqdm import tqdm

def load_dataset(dataset_path):
    if dataset_path.endswith(".json"):
        with open(dataset_path, "r") as f:
            dataset = json.load(f)
            if type(dataset) == dict:
                dataset = dataset.get("Instances")
            print(f"Loaded {len(dataset)} instances")
        return dataset
    else:
        raise ValueError("Dataset format not supported")

def save_json(data, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_model_tokenizer(model_name_or_path, adapter_path=None):
    """
    :param model_name_or_path: str, model name or path to the model
    :return: model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer
    """
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, torch_dtype=torch.bfloat16, device_map="auto", cache_dir="./cache", )
    if adapter_path is not None:
        print("Using LoRA model")
        print(f"Loading adapter from {adapter_path}")
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.bfloat16, device_map="auto")
    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    
    return model, tokenizer, generation_config

def load_model_tokenizer_init_left(model_name_or_path, adapter_path=None):
    """
    :param model_name_or_path: str, model name or path to the model
    :return: model: transformers.PreTrainedModel, tokenizer: transformers.PreTrainedTokenizer
    """
    print("Left")
    config = AutoConfig.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, padding_side = "left")
    print(tokenizer.padding_side)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, torch_dtype=torch.bfloat16, device_map="auto", cache_dir="./cache", )
    if adapter_path is not None:
        print("Using LoRA model")
        print(f"Loading adapter from {adapter_path}")
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, adapter_path, torch_dtype=torch.bfloat16, device_map="auto")
    generation_config = GenerationConfig.from_pretrained(model_name_or_path)
    
    return model, tokenizer, generation_config


def generate_response_batch(model, tokenizer, prompts: list, sys_prompt="", batch_size=32, generation_config=None, few_shot="") -> list:
    """
    :param model: transformers.PreTrainedModel
    :param tokenizer: transformers.PreTrainedTokenizer
    :param prompts: list of str, user inputs
    :param max_new_tokens: int, max_new_tokens of the generated response
    :param sys_prompt: str, system prompt
    :return: responses: list of str, generated responses
    """
    support_system_prompt = True
    try:
        test_message = [
            {
                "role": "system",
                "content": "test",
            }
        ]
        tokenizer.apply_chat_template(test_message, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        support_system_prompt = False
    
    if support_system_prompt:
        init_message = [
            {
                "role": "system",
                "content": sys_prompt,
            },
        ]
    else:
        init_message = []
    if few_shot:
        with open(few_shot, "r") as f:
            few_shot_data = json.load(f)[:5]
        for instance in few_shot_data:
            init_message += [
                {
                    "role": "user",
                    "content": instance['input']
                },
                {
                    "role": "assistant",
                    "content": instance['output']
                }  
            ]        
    if type(prompts[0]) == str:
        messages = [
            init_message + [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            for prompt in prompts
        ]
    else:
        messages = [
            init_message + prompt
            for prompt in prompts
        ]
    all_responses = []
    for i in tqdm(range(0, len(messages), batch_size)):
        inputs = tokenizer.apply_chat_template(messages[i:i+batch_size], tokenize=False, add_generation_prompt=True)
        print(inputs[0])
        max_len = max(len(tokenizer.encode(q)) for q in inputs)
        print(f'max_len: {max_len}')
        inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model.device)
            
        output = model.generate(**inputs, generation_config=generation_config)
        
        responses = [tokenizer.decode(output[j][len(inputs['input_ids'][j]):], skip_special_tokens=True) for j in range(len(inputs['input_ids']))]
        all_responses += responses
    
    return all_responses