#!/bin/sh

python -u merge_llama.py

# 설정 배열 정의
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

# 각 설정에 대해 반복
for config in "${configs[@]}"; do
  # 공백을 기준으로 설정 분리
  read -r r alpha epoch seed <<< "$config"
  
  echo "Processing: r=$r, alpha=$alpha, epoch=$epoch, seed=$seed"
  
  # Python 스크립트 실행
  python -u merge_llama.py $r $alpha $epoch $seed
  
  echo "--------------------"
done