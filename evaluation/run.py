import os
import time
from fire import Fire
from metrics import *
from utils import *

ds2metric = {
    "HEx-PHI": "ASR",
    "BeaverTails": "ASR",
    "AdvBench": "ASR",
    "aim": "ASR",
    "Autodan": "ASR",
    "GCG": "ASR",
    "Cipher": "ASR",
    "Codechameleon": "ASR",
    "RoleBench_RAW": "RougeL",
    "RoleBench_SPE": "RougeL",
}

id2model = {
    "llama3": "Meta-Llama-3-8B-Instruct",
    "llama3.1": "Llama-3.1-8B-Instruct-hf",
}

def main(
    decoding_result_path: str,
    dataset_name: str,
    save_tag: str="",
    eval_result_dir: str="SaRFT/evaluation/results",
):
    if save_tag == "":
        save_tag = os.path.basename(decoding_result_path).replace(".json", "")
        print(f"save_tag: {save_tag}")
    os.makedirs(eval_result_dir, exist_ok=True)
    date = time.strftime("%Y-%m-%d", time.localtime())
    save_path = os.path.join(eval_result_dir, dataset_name, date, f"{save_tag}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    decoding_results = load_json(decoding_result_path)
    _metric = eval(ds2metric[dataset_name])

    metric = _metric()
    eval_results = metric(decoding_results)
    save_json(eval_results, save_path)
    if "data" in eval_results:
        eval_results.pop("data")

    print("### Evaluation Results ###")
    print(f"Dataset: {dataset_name}")
    print(f"Metric: {ds2metric[dataset_name]}")
    print(f"Save Path: {save_path}")
    print(eval_results)

if __name__ == "__main__":
    Fire(main)