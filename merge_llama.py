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
