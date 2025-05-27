# Copyright (C) 2023 ByteDance. All Rights Reserved.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import random
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling

torch.manual_seed(8888)
np.random.seed(8888)
random.seed(8888)

def create_pku_dataloader_from_dataset(tokenizer, dataset, fraction=1.0, batch_size=4):
    """
    Given the PKU dataset, create the dataloader on the unlearned harmful Q&A pairs.

    Args:
        tokenizer: Tokenizer.
        dataset: Loaded PKU dataset.
        fraction: <1 will do downsampling.
        batch_size: Batch size.

    Returns:
        Data loader of PKU harmful Q&A pairs.
    """

    # Preproccess function.
    def preproccess(examples):
        """
        Input: Dict[List]
        Output: Dict[List]
        """
        results = {"input_ids": [], "attention_mask": [], "start_locs": []}

        for i in range(len(examples["prompt"])):
            # Subsample if needed.
            if random.random() > fraction:
                continue

            prompt = examples["prompt"][i]
            response_list = []

            # Add only bad samples.
            if not examples["is_response_0_safe"][i]:
                response_list.append(examples["response_0"][i])
            if not examples["is_response_1_safe"][i]:
                response_list.append(examples["response_1"][i])

            # Add all responses to results or skip if none.
            for response in response_list:
                text = f"<s> Input: {prompt}\nOutput: \n{response}"
                tokenized = tokenizer(text, truncation=True, padding="max_length")
                results["input_ids"].append(tokenized["input_ids"])
                results["attention_mask"].append(tokenized["attention_mask"])
                # Calculate start idx for answer
                # test_text = f"### Question: {prompt}\n ### Answer: "
                test_text = f"<s> Input: {prompt}\nOutput: \n"
                test_tokenized = tokenizer(
                    test_text, truncation=True, padding="max_length"
                )
                results["start_locs"].append(len(test_tokenized["input_ids"]) - 1)

        return results

    # Need to drop all original columns to emit more than one row for each original row https://huggingface.co/docs/datasets/about_map_batch#input-size-output-size.
    dataset = dataset.map(
        preproccess,
        batched=True,
        remove_columns=[
            "prompt",
            "response_0",
            "response_1",
            "is_response_0_safe",
            "is_response_1_safe",
            "better_response_id",
            "safer_response_id",
        ],
    )
    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "start_locs"]
    )

    # Add labels and make it data loader.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=data_collator
    )

    return dataloader


def create_truthfulqa_dataloader(tokenizer, batch_size=4):
    """
    Create the TruthfulQA dataloader for the normal data.

    Args:
        tokenizer: Tokenizer.
        batch_size: Batch size.

    Returns:
        Data loader of TruthfulQA normal Q&A pairs.
    """
    df = pd.read_csv("data/TruthfulQA.csv")
    questions, good_answers = df["Question"].values, df["Best Answer"].values

    data = {"input_ids": [], "attention_mask": []}
    for question, good_answer in zip(questions, good_answers):
        # text = f"### Question: {question}\n ### Answer: {good_answer}"
        text = f"<s> Input: {question}\nOutput: \n{good_answer}"
        tokenized = tokenizer(text, truncation=True, padding="max_length")
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])

    dataset = Dataset.from_dict(data)

    # Split train/val/test = 0.7/0.1/0.2.
    train_len = int(0.7 * len(dataset))
    val_len = int(0.1 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, collate_fn=data_collator, shuffle=True
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_truthfulQA_answers_plaintext(tqa_file_path="/home/ylhu/CodeSpace/24/EMNLP_24/BenchMark_Analysis/datasets/train/TruthfulQA/TruthfulQA.csv"):
    """
    Get the plain text of TruthfulQA's answers used for random mismatch.

    Args:
        None

    Returns:
        A list of answer text in TruthfulQA.
    """
    ans_names = ["Best Answer", "Correct Answers", "Incorrect Answers"]

    df = pd.read_csv(tqa_file_path)
    all_ans = []
    for ans_name in ans_names:
        answers = df[ans_name].values
        if ans_name == "Best Answer":
            all_ans.extend(answers)
        # Split "Correct Answers" and "Incorrect Answers"by ";".
        else:
            for answer in answers:
                ans_list = answer.split(";")
                for ans in ans_list:
                    all_ans.append(ans.strip())

    return all_ans

def get_pku_answers_plaintext(pku_file_path="/home/ylhu/CodeSpace/24/EMNLP_24/BenchMark_Analysis/datasets/train/BeaverTails/train.json"):
    """
    Get the plain text of TruthfulQA's answers used for random mismatch.

    Args:
        None

    Returns:
        A list of answer text in TruthfulQA.
    """
    with open(pku_file_path, encoding="utf-8") as task_f:
            s = task_f.read()
            instances = json.loads(s)
    
    all_ans = []
    for item in instances['Instances']:
        all_ans.append(item['output'])

    return all_ans


def compute_kl(pretrained_model, current_model, batch, device):
    """
    Compute *forward* KL as the normal utility loss.

    Args:
        pretrained_model: reference model which is the pretrained (original) model.
        current_model: The current unlearning model.
        batch: A batch of normal data.
        device: GPU device.

    Returns:
       The KL loss.
    """
    normal_outputs = current_model(
        batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )

    with torch.no_grad():
        pretrained_outputs = pretrained_model(
            batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )

    # P: pretrained model; Q: current model.
    prob_p = torch.nn.functional.softmax(pretrained_outputs.logits, -1)
    prob_q = torch.nn.functional.softmax(normal_outputs.logits, -1)

    # loss = (prob_p * torch.log(prob_q + 1e-12)).sum(-1).mean()
    loss = F.kl_div(prob_q.log(), prob_p, reduction="batchmean")

    return loss


def get_answer_loss(operation, batch, model, device="cuda:0"):
    """
    Compute the loss on the answer (i.e. y) part.

    Args:
        operation: either "ga" (gradient ascent) or "gd" (gradient descent).
        batch: A batch of data.
        model: The unlearned model.
        device: GPU device.

    Returns:
       The loss.
    """
    assert operation in ["ga", "gd"], "Operation must be either GA or GD."
    input_ids, attention_mask, start_locs, labels = (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        batch["start_locs"],
        batch["labels"].to(device),
    )
    outputs = model(input_ids, attention_mask=attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    # Shift one to predict next token.
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    losses = []
    for bid in range(input_ids.shape[0]):
        one_inp, one_st = input_ids[bid], start_locs[bid]

        # GA or GD.
        position_loss = loss_fct(shift_logits[bid], shift_labels[bid])
        if operation == "ga":  # Negative the direction for GA.
            position_loss = -position_loss

        # Simply put equal weights on all answers.
        position_weight = torch.zeros_like(one_inp)
        assert len(position_weight) == len(position_loss) + 1
        position_weight[one_st:] = 1  # only focus on answer part

        # Ignore the padding part.
        position_weight[one_inp == 1] = 0
        if position_weight.sum() > 0:
            position_weight = position_weight / position_weight.sum()

        one_loss = (position_weight[:-1] * position_loss).sum()
        losses.append(one_loss)
    final_loss = torch.stack(losses).mean()

    return final_loss


def get_rand_ans_loss(bad_batch, tokenizer, normal_ans, model, K=5, device="cuda:0"):
    """
    Compute the loss of the random mismatch.

    Args:
        bad_batch: A batch of forgetting data.
        tokenizer: The tokenizer.
        normal_ans: A list of random answers.
        model: unlearned model.
        K: How many random answers sampled for each forgetting sample.
        device: GPU device.

    Returns:
       The random mismatch loss.
    """
    bad_input_ids = bad_batch["input_ids"].to(device)
    rand_ans_list = random.sample(normal_ans, k=K)

    batch_random_features = []
    for batch_idx in range(bad_input_ids.shape[0]):
        single_input_id = bad_input_ids[batch_idx, :]
        ori_text = tokenizer.decode(single_input_id)
        # Get question.

        # question = ori_text.split("###")[1].split("Question:")[-1].strip()
        # question_prefix = f"### Question: {question}\n### Answer: "
        question = ori_text.split("Input: ")[1].split("\nOutput:")[0].strip()
        question_prefix = f"<s> Input: {question}\nOutput: \n"
        tokenized_question_prefix = tokenizer(
            question_prefix, truncation=True, padding="max_length"
        )
        # Doesn't need to minus 1 because there's a starting token in the beginning.
        start_loc = len(tokenized_question_prefix)

        # Get random answer.
        for rand_ans in rand_ans_list:
            random_sample = f"{question_prefix}{rand_ans}"
            # Tokenize.
            tokenized_rs = tokenizer(
                random_sample, truncation=True, padding="max_length"
            )
            batch_random_features.append(
                {
                    "input_ids": tokenized_rs["input_ids"],
                    "attention_mask": tokenized_rs["attention_mask"],
                    "start_locs": start_loc,
                }
            )

    # Batchify.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    batch_random = data_collator(batch_random_features)

    # GD on answer.
    random_loss = get_answer_loss("gd", batch_random, model, device=device)

    return random_loss

def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []
    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None 
    
    hook_handle = module.register_forward_hook(hook)
    
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)
        
    hook_handle.remove()

    return cache[0]

def compute_forget_loss(updated_model, forget_inputs, updated_module, steering_coeff):    
    updated_forget_activations = forward_with_cache(
        updated_model, forget_inputs, module=updated_module, no_grad=False
    ).to(updated_model.device)

    random_vector = torch.rand_like(updated_forget_activations, dtype=updated_model.dtype, device=updated_model.device)
    control_vec = random_vector / torch.norm(random_vector) * steering_coeff
    
    forget_loss = torch.nn.functional.mse_loss(
        updated_forget_activations, control_vec
    )
    
    return forget_loss

def compute_retain_loss(updated_model, frozen_model, retain_inputs, updated_module, frozen_module):
    updated_retain_activations = forward_with_cache(
        updated_model, retain_inputs, module=updated_module, no_grad=False
    ).to(updated_model.device)

    frozen_retain_activations = forward_with_cache(
        frozen_model, retain_inputs, module=frozen_module, no_grad=True
    ).to(updated_model.device)

    retain_loss = torch.nn.functional.mse_loss(
        updated_retain_activations, frozen_retain_activations
    )
    
    return retain_loss

import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig, pipeline, set_seed
from peft import PeftModel, PeftConfig
from tqdm import tqdm

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
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, torch_dtype=torch.bfloat16, device_map="auto")
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