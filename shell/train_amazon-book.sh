#!/bin/sh

seed=425
base_model='meta-llama/Meta-Llama-3-8B-Instruct'
dataset='amazon-book'
batch_size=32
micro_batch_size=4
epochs=3
lr=2e-4
cutoff_len=1200
patience=1
lora_r=128
lora_alpha=32
dropout=0.1

echo "dataset: $dataset, seed: $seed, lora_r: $lora_r, lora_alpha: $lora_alpha, lora_dropout: $dropout"
NCCL_P2P_DISABLE=1 accelerate launch \
    --config_file ./shell/accelerate_config.yaml train.py \
    --base_model $base_model \
    --dataset $dataset \
    --train_data_path "data/${dataset}/rating_bias/train.json" \
    --val_data_path "data/${dataset}/rating_bias/valid.json" \
    --output_dir "${dataset}_r${lora_r}_alpha${lora_alpha}_seed${seed}" \
    --batch_size $batch_size \
    --micro_batch_size $micro_batch_size \
    --num_epochs $epochs \
    --learning_rate $lr \
    --cutoff_len $cutoff_len \
    --rmse_patience $rmse_patience \
    --lora_r $lora_r \
    --lora_alpha $lora_alpha \
    --lora_dropout $dropout \
    --group_by_length \
    --seed $seed 
    # --resume_from_checkpoint \
    # --lora_weights "/home/jieyong99/EXP3RT/imdb_trained_seed42/checkpoints/checkpoint-epoch10"
