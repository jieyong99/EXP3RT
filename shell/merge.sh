#!/bin/sh

python -u merge_llama.py


declare -a configs=(
  # "64 16 2 42"
  # "64 16 4 42"
  # "128 16 2 42"
  # "128 16 3 42"
  # "128 16 4 42"
  # "128 256 2 42"
  # "128 256 4 42"
  # "256 16 2 42"
  "512 128 2 42"
  # "256 16 6 42"
)


for config in "${configs[@]}"; do

  read -r r alpha epoch seed <<< "$config"
  
  echo "Processing: r=$r, alpha=$alpha, epoch=$epoch, seed=$seed"
  
  python -u merge_llama.py $r $alpha $epoch $seed
  
  echo "--------------------"
done
