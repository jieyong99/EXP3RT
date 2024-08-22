TASK=generate_preference # generate_user_side / generate_item_side / generate_reasoning
python ./generate.py \
    --api_owner \
    --task $TASK \
    --input_path \
    --prompt_path ./prompt/preference_extraction.txt \
    --save_dir \
    --model_name gpt-3.5-turbo-1106 \
    --temperature 0.1 \
    # --max_tokens 500