import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import sys
import fire
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import re
import shutil
import random
from transformers import LlamaForCausalLM, LlamaTokenizer
import random
import json
import numpy as np
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
##
from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
from datasets import load_dataset, Dataset

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


os.environ['WANDB_DISABLED'] = 'true'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    dataset: str = "",
    train_data_path: str = "",
    val_data_path: str = "",
    output_dir: str = "",
    seed: int = 0,
    # training hyperparams
    batch_size: int = 64,
    micro_batch_size: int = 16,
    num_epochs: int = 10,
    learning_rate: float = 3e-5,
    cutoff_len: int = 1024,
    rmse_patience: int = 5,
    # lora hyperparams
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    # others
    group_by_length: bool = True,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: bool = False,  # either training checkpoint or final adapter
    lora_weights: str = "",
):
    print(
        f"\nTraining Llama3-8B + QLoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"train_data_path: {train_data_path}\n"
        f"val_data_path: {val_data_path}\n"
        f"dataset: {dataset}\n"
        f"output_dir: {output_dir}\n"
        f"batch_size: {batch_size}\n"
        f"micro_batch_size: {micro_batch_size}\n"
        f"num_epochs: {num_epochs}\n"
        f"learning_rate: {learning_rate}\n"
        f"cutoff_len: {cutoff_len}\n"
        f"lora_r: {lora_r}\n"
        f"lora_alpha: {lora_alpha}\n"
        f"lora_dropout: {lora_dropout}\n"
        f"group_by_length: {group_by_length}\n"
        f"resume_from_checkpoint: {resume_from_checkpoint}\n"
    )
    assert base_model, "Please specify a --base_model, e.g. --base_model='meta-llama/Meta-Llama-3-8B-Instruct'"
        
    gradient_accumulation_steps = batch_size // micro_batch_size
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # tokenizer = AutoTokenizer.from_pretrained(lora_weights)
    # model = AutoModelForCausalLM.from_pretrained(lora_weights)

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype='bfloat16',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        ),
        torch_dtype=torch.bfloat16,
    )

    def is_ipex_available():
        def get_major_and_minor_from_version(full_version):
            return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

        _torch_version = importlib.metadata.version("torch")
        if importlib.util.find_spec("intel_extension_for_pytorch") is None:
            return False
        _ipex_version = "N/A"
        try:
            _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
        except importlib.metadata.PackageNotFoundError:
            return False
        torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
        ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
        if torch_major_and_minor != ipex_major_and_minor:
            warnings.warn(
                f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
                f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
            )
            return False
        return True

    compute_dtype = torch.bfloat16
    if compute_dtype == torch.float16:
        if torch.cuda.is_bf16_supported():
            print("=" * 80)
            print("Your GPU supports bfloat16, you can accelerate training with the argument --bf16")
            print("=" * 80)
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print("Intel XPU does not support float16 yet, so switching to bfloat16")

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)
    model.config.torch_dtype = torch.bfloat16

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    def find_all_linear_names(model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if "lm_head" in lora_module_names:  # needed for 16-bit
            lora_module_names.remove("lm_head")
        return list(lora_module_names)


    modules = find_all_linear_names(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"  # Allow batched inference
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
        
    def generate_and_tokenize_prompt(data_point):
        if dataset == "imdb":
            s_item = "movie"
            l_item = "Movie"
            
            input_text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"These are users' preferences about the movie : {data_point['input']}\n"
                "Based on this preferences, point out the "preference" people liked and disliked about the item under [Like] and [Dislike] in bullet point, respectively."
                "If there is nothing to mention about like/dislike, simply write "None." under the corresponding tag.\n"        
                
                "### Output Format:\n"
                "[Like]"
                "- Encapsulate the "preference" people liked about the item in bullet points.\n"
                "[Dislike]"
                "- Encapsulate the "preference" people disliked about the item in bullet points.<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )

            preference = data_point['output']

            output_text = (
                f"Preference: {preference}<|eot_id|><|end_of_text|>"
            )
            
        elif dataset == "amazon-book":
            s_item = "book"
            l_item = "Book"
        
            input_text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"These are users' preferences about the book : {data_point['input']}\n"
                "Based on this preferences, point out the "preference" people liked and disliked about the item under [Like] and [Dislike] in bullet point, respectively."
                "If there is nothing to mention about like/dislike, simply write "None." under the corresponding tag.\n"        
                
                "### Output Format:\n"
                "[Like]"
                "- Encapsulate the "preference" people liked about the item in bullet points.\n"
                "[Dislike]"
                "- Encapsulate the "preference" people disliked about the item in bullet points.<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )

            preference = data_point['output']

            output_text = (
                f"Preference: {preference}<|eot_id|><|end_of_text|>"
            )
            
        full_prompt = f"{input_text}\n{output_text}"
        
        
        # Tokenize input text to get its length
        tokenized_prompt = tokenizer(
            full_prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )

        # Tokenize input text to get its length
        tokenized_input = tokenizer(
            input_text,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )
        
        tokenized_prompt["labels"] = tokenized_prompt["input_ids"].copy()
        
        user_prompt_len = len(tokenized_input["input_ids"])
        
        tokenized_prompt["labels"] = [-100] * user_prompt_len + tokenized_prompt["labels"][user_prompt_len:]

        return tokenized_prompt


    if train_data_path.endswith(".json"):
        train_data = load_dataset("json", data_files=train_data_path)
    else:
        train_data = load_dataset(train_data_path)
    
    if val_data_path.endswith(".json"):
        val_data = load_dataset("json", data_files=val_data_path)
    else:
        val_data = load_dataset(val_data_path)

    model.print_trainable_parameters()

    train_data = train_data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
    val_data = val_data["train"].shuffle(seed=seed).map(generate_and_tokenize_prompt)
 

    save_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    custom_callback = SaveBestModelCallback(save_dir=save_dir, output_dir=output_dir, model=model, tokenizer=tokenizer)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=0.03,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=10,
            weight_decay=0.0,
            lr_scheduler_type="constant",
            max_grad_norm=0.3,
            evaluation_strategy="epoch",
            save_strategy="no",
            output_dir=output_dir,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            gradient_checkpointing=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),

        callbacks=[custom_callback],  
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
        
    trainer.train()
    
    
    tokenizer_to_save = AutoTokenizer.from_pretrained(output_dir)

    base_model_for_save = AutoModelForCausalLM.from_pretrained(
            base_model, return_dict=True, torch_dtype=torch.bfloat16
        )

    model_to_save = PeftModel.from_pretrained(base_model_for_save, output_dir)
    model_to_save = model_to_save.merge_and_unload()
    
    merged_dir = os.path.join(output_dir, 'merged')

    os.makedirs(merged_dir, exist_ok=True)
    model_to_save.save_pretrained(merged_dir)
    tokenizer_to_save.save_pretrained(merged_dir)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n If there's a warning about missing keys above, please disregard :)")


if __name__ == "__main__":
    fire.Fire(train)
