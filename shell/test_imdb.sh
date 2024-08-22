#!/bin/sh

seed=42
base_model='meta-llama/Meta-Llama-3-8B-Instruct'
dataset='imdb'
lora_r=512
lora_alpha=128
data_type='test_helpful'
# epoch=2

# echo "dataset: $dataset, seed: $seed, lora_r: $lora_r, lora_alpha: $lora_alpha, saved_epoch: $epoch, TEST WITH VLLM!!"
echo "dataset: $dataset, data_type: $data_type, seed: $seed, lora_r: $lora_r, lora_alpha: $lora_alpha, TEST WITH VLLM!!"
CUDA_VISIBLE_DEVICES=0 python -u test.py \
    --base_model $base_model \
    --dataset $dataset \
    --test_data_path "data/${dataset}/rating_bias/${data_type}.json" \
    --output_dir "${dataset}_${data_type}_r${lora_r}_alpha${lora_alpha}_seed${seed}_result.json" \
    --seed $seed \
    --saved_path "${dataset}_r${lora_r}_alpha${lora_alpha}_seed${seed}/merged"

