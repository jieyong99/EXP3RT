# import os
# from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
# import sys
# from typing import List
# from heapq import heappush, heappop
# import fire
# import torch
# import transformers
# from datasets import load_dataset
# from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
# import re
# import shutil
# import random
# from transformers import LlamaForCausalLM, LlamaTokenizer
# import random
# import json
# import numpy as np
# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
#     prepare_model_for_int8_training,
#     set_peft_model_state_dict,
# )
# ##
# from collections import defaultdict, OrderedDict
# import copy
# import json
# import os
# from os.path import exists, join, isdir
# from dataclasses import dataclass, field
# import sys
# from typing import Optional, Dict, Sequence
# import numpy as np
# from tqdm import tqdm
# import logging
# import bitsandbytes as bnb
# import pandas as pd
# import importlib
# from packaging import version
# from packaging.version import parse

# import torch
# import transformers
# from torch.nn.utils.rnn import pad_sequence
# import argparse
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     set_seed,
#     Seq2SeqTrainer,
#     BitsAndBytesConfig,
#     LlamaTokenizer,
# )

# from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel

# # def merge_llama(peft_path, output_path):
# #     # import shutil
# #     max_step = 0
# #     for item in os.listdir(peft_path):
# #         if os.path.isdir(os.path.join(peft_path, item)) and item.startswith("checkpoint"):
# #             max_step = max(max_step, int(item.replace("checkpoint-", "")))
# #     peft_path = os.path.join(peft_path, f'checkpoint-{max_step}')
# #     base_model = LlamaForCausalLM.from_pretrained(
# #         'meta-llama/Llama-2-7b-hf', return_dict=True, torch_dtype=torch.float16
# #     )
# #     tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
# #     model = PeftModel.from_pretrained(base_model, peft_path)
# #     model = model.merge_and_unload()
# #     #first attempt
# #     os.makedirs(output_path, exist_ok=True)
# #     #second attempt
# #     # try:
# #     #     # Deletes all files and subdirectories under 'directory'
# #     #     shutil.rmtree(output_path)
# #     #     print(f"All files and subdirectories in '{output_path}' have been deleted successfully.")
# #     # except OSError as e:
# #     #     print(f"Error: {output_path} : {e.strerror}")
# #     model.save_pretrained(output_path)
# #     tokenizer.save_pretrained(output_path)
    
# print(f"file path: imdb_r64_alpha16_seed42/checkpoints/checkpoint-epoch4")
    
# tokenizer = AutoTokenizer.from_pretrained('imdb_r64_alpha16_seed42/checkpoints/checkpoint-epoch4')

# base_model = AutoModelForCausalLM.from_pretrained(
#         'meta-llama/Meta-Llama-3-8B-Instruct', return_dict=True, torch_dtype=torch.bfloat16
#     )


# model = PeftModel.from_pretrained(base_model, 'imdb_r64_alpha16_seed42/checkpoints/checkpoint-epoch4')
# model = model.merge_and_unload()

# os.makedirs('imdb_r64_alpha16_seed42/merged', exist_ok=True)
# model.save_pretrained('imdb_r64_alpha16_seed42/merged')
# tokenizer.save_pretrained('imdb_r64_alpha16_seed42/merged')

import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def process_model(r, alpha, epoch, seed):
    # base_path = f"imdb_r{r}_alpha{alpha}_seed{seed}"
    # checkpoint_path = f"{base_path}/checkpoints/checkpoint-epoch{epoch}"
    # merged_path = f"{base_path}/merged_epoch{epoch}"
    
    checkpoint_path = 'amazon-book_no_reasoning_r128_alpha32_seed42'
    merged_path = 'amazon-book_no_reasoning_r128_alpha32_seed42/merged'

    # print(f"Processing: r={r}, alpha={alpha}, epoch={epoch}, seed={seed}")
    print(f"Checkpoint path: {checkpoint_path}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        'meta-llama/Meta-Llama-3-8B-Instruct', return_dict=True, torch_dtype=torch.bfloat16
    )

    model = PeftModel.from_pretrained(base_model, checkpoint_path)

    os.makedirs(merged_path, exist_ok=True)
    model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)

    print(f"Model saved to: {merged_path}")
    print("--------------------")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python merge_llama.py <r> <alpha> <epoch> <seed>")
        sys.exit(1)
    
    r = int(sys.argv[1])
    alpha = int(sys.argv[2])
    epoch = int(sys.argv[3])
    seed = int(sys.argv[4])

    process_model(r=r, alpha=alpha, epoch=epoch, seed=seed)