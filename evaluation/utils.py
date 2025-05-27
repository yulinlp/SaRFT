import json
from transformers import AutoTokenizer

def load_json(file_path: str):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_tokenizer(model_name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    return tokenizer