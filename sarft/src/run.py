"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional
import torch

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pickle
from datasets import load_dataset, DatasetDict
import transformers
# from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    AutoModelForCausalLM,
    set_seed, )
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig # add
from collator import DataCollator
# import loralib as lora
from trainer import Trainer, skip_instructions
from compute_metrics import compute_metrics

import time
import subprocess
import threading
def monitor_gpu_memory(interval=1):
    while True:
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print("###")

        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        print(result.stdout)
        print(f"Allocated Memory: {allocated / (1024 ** 2):.2f} MB")
        print(f"Reserved Memory: {reserved / (1024 ** 2):.2f} MB")
        time.sleep(interval)
        
# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

local_data_path = "/home/work/nltk_data"
nltk.data.path.append(local_data_path)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    # added for AutoCL
    lora_dim: Optional[int] = field(
        default=16,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )

    prefix_len: Optional[int] = field(
        default=10,
        metadata={
            "help": "Length of Prompt."
        },
    )

    mlp_hidden_dim: Optional[int] = field(
        default=800,
        metadata={
            "help": "Intrinsic dimension of the latent MLP space."
        },
    )

    attn_temperature: Optional[int] = field(
        default=1,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    lora_r: Optional[int] = field(
        default=32,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    lora_alpha: Optional[int] = field(
        default=64,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )

    run_single: bool = field(
        default=False,
        metadata={
            "help": "Temperature to control attention weights."
        },
    )

    previous_lora_path: Optional[str] = field(
        default=None,
        metadata={"help": "the path to load previous prompts."}
    )

    previous_prompt_key_path: Optional[str] = field(
        default=None,
        metadata={"help": "the path to load previous prompts."}
    )

    load_checkpoint_from: str = field(
        default=None,
        metadata={"help": "Path to load previous checkpoints"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    good_data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the generated train/dev/test splits."}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    do_lora: bool = field(
        default=False,
        metadata={"help": "whether to train with lora"}
    )
    sys_prompt: Optional[str] = field(
        default="",
        metadata={"help": ""}
    )
    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "beta for kl divergence"}
    )
    reward_type: Optional[str] = field(
        default="",
        metadata={"help": "reward type for reinforcement learning"}
    )
    gradiant_update: Optional[str] = field(
        default="gd",
        metadata={"help": "gradiant update for reinforcement learning"}
    )
    safe_loss_type: Optional[str] = field(
        default="kl",
        metadata={"help": "safe loss type for reinforcement learning"}
    )

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args._frozen = False
    training_args.beta = data_args.beta
    training_args.reward_type = data_args.reward_type
    training_args.gradiant_update = data_args.gradiant_update
    training_args.safe_loss_type = data_args.safe_loss_type

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "dataset.py"),
        data_dir=data_args.data_dir,
        trust_remote_code=True
    )
    
    raw_datasets.cleanup_cache_files()
    print(raw_datasets)
    
    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir = model_args.cache_dir,
        use_fast = model_args.use_fast_tokenizer,
        revision = model_args.model_revision,
        use_auth_token = True if model_args.use_auth_token else None,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        use_safetensors=True,
        torch_dtype=torch.bfloat16
    )
    
    print(model)

    if data_args.do_lora:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_r, lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, target_modules=["q_proj", "v_proj"])
        model = get_peft_model(model, peft_config)
    
    model.resize_token_embeddings(len(tokenizer))

    if data_args.do_lora:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "lora" in name:
                param.requires_grad = True
    
    # Calculate n_params    
    total_params, params = 0, 0
    for n, p in model.named_parameters():
        if p.requires_grad:
        # if any([x in n for x in ["router", "A", "z"]]):
            total_params += p.numel()
        params += p.numel()


    print(
        "Total number of parameters: {}M, rate: {}%".format(
            total_params // 1000 / 1000, round(total_params / params * 100, 2)
        )
    )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollator(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file,
        sys_prompt=data_args.sys_prompt
    )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    # Metric
    def compute_rouge_metrics(dataset, preds, model, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer, answer_prefix=data_args.answer_prefix)
        references = [e["Instance"]["label"] for e in dataset]
        sources = [e["Instance"]['sentence'] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references, tokenizer=tokenizer, input_ids=None, model=model)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Prediction": pred
                    }) + "\n")
        return result
    
    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
    )
    
    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        
        save_path = training_args.output_dir + "/saved_weights"
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
            except:
                pass

        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        
        # save rewards
        rewards = trainer.rewards
        with open(os.path.join(training_args.output_dir, "train_rewards.json"), 'w') as f:
            json.dump(rewards, f)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

if __name__ == "__main__":
    main()
