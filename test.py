import fire
from datasets import load_dataset
import json
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset

from vllm import LLM, SamplingParams

import os
import sys
import fire
import torch
import transformers
from datasets import load_dataset
import random
##
import copy
import json
import numpy as np
import logging
import bitsandbytes as bnb
import pandas as pd
from packaging import version
from packaging.version import parse

import torch
import transformers


os.environ['WANDB_DISABLED'] = 'true'
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def softmax(x, temp=1.0):
    exp_x = np.exp((x - np.max(x)) / temp)
    return exp_x / exp_x.sum()

def is_valid_digit(token, dataset):
    try:
        num = int(token) if isinstance(token, str) else token
        if dataset == "imdb":
            return 0 <= num <= 9
        elif dataset == "amazon-book":
            return 1 <= num <= 5
        return False
    except ValueError:
        return False

def sort_probabilities(digit_logprobs, probs, dataset):
    if dataset == "imdb":
        all_digits = range(10)
    elif dataset == "amazon-book":
        all_digits = range(1, 6)
    
    sorted_probs = {str(i): 0.0 for i in all_digits}
    for k, p in zip(digit_logprobs.keys(), probs):
        sorted_probs[k] = float(p)
    
    return sorted_probs

def test(
    # model/data params
    base_model: str = "",  # the only required argument
    dataset: str = "",
    test_data_path: str = "",
    output_dir: str = "",
    seed: int=42,
    saved_path: str="",
):
    print(
        f"\nTraining Llama3-8B + QLoRA model with params:\n"
        f"base_model: {base_model}\n"
        f"testdata_path: {test_data_path}\n"
        f"dataset: {dataset}\n"
        f"output_dir: {output_dir}\n"
        f"seed: {seed}\n"
        f"saved_path: {saved_path}\n"
    )
    assert base_model, "Please specify a --base_model, e.g. --base_model='meta-llama/Meta-Llama-3-8B-Instruct'"

    def generate_and_tokenize_prompt(data_point):
        if dataset == "imdb":
            def imdb_process_user_rating(text):
                parts = text.split("[Average Rating]")
                if len(parts) > 1:
                    try:
                        rating = float(parts[-1].strip())
                        new_rating = max(0, rating - 1) 
                        return f"{parts[0]}[User Average Rating]\n{new_rating:.1f}"
                    except ValueError:
                        return text
                return text
            def imdb_process_item_rating(text):
                parts = text.split("[Average Rating]")
                if len(parts) > 1:
                    try:
                        rating = float(parts[-1].strip())
                        new_rating = max(0, rating - 1)  
                        return f"{parts[0]}[Item Average Rating]\n{new_rating:.1f}"
                    except ValueError:
                        return text
                return text

            data_point['user_persona'] = imdb_process_user_rating(data_point['user_persona'])
            data_point['item_synopsis'] = imdb_process_item_rating(data_point['item_synopsis'])
            
            s_item = "movie"
            l_item = "Movie"
            
            input_text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"You are a helpful AI assistant for {s_item} recommendation. Based on the user's preferences and {s_item} characteristics provided, generate a recommendation reasoning and predict the user's rating.\n"
                "You must always generate a response in the following format whenever the user provides information:\n"
                f"Reasoning: [Provide a detailed, single-paragraph reasoning for your prediction, addressing at least three specific points of alignment or misalignment between the user's preferences and the {s_item}'s characteristics.]\n"
                f"Predicted User Rating: [Predict the user's rating as an integer from 0 to 9: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. 0 indicates the user would strongly dislike the {s_item}, while 9 indicates the user would highly enjoy and recommend it. Consider the average ratings provided for the user and the {s_item} in your prediction.]\n"
                "Note: Do not simply repeat the input text. Generate a new reasoning and rating prediction based on the input provided.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>"
                f"I need a recommendation for this {s_item}. Here's the information:\n"
                "User Preferences:\n"
                "<User Persona>\n"
                f"{data_point['user_persona']}\n\n"
                f"{l_item} Characteristics:\n"
                f"<{l_item} Description>\n"
                f"{data_point['item_description']}\n\n"
                f"<{l_item} Synopsis>\n"
                f"{data_point['item_synopsis']}\n\n"
                f"Based on this information, please provide a detailed reasoning for your recommendation and predict a rating for this {s_item}. Follow the format specified in the system instructions.<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
            
        elif dataset == "amazon-book":
            def process_user_rating(text):
                parts = text.split("[Average Rating]")
                if len(parts) > 1:
                    try:
                        rating = float(parts[-1].strip())
                        return f"{parts[0]}[User Average Rating]\n{rating:.1f}"
                    except ValueError:
                        return text
                return text
            def process_item_rating(text):
                parts = text.split("[Average Rating]")
                if len(parts) > 1:
                    try:
                        rating = float(parts[-1].strip()) 
                        return f"{parts[0]}[Item Average Rating]\n{rating:.1f}"
                    except ValueError:
                        return text
                return text

            data_point['user_persona'] = process_user_rating(data_point['user_persona'])
            data_point['item_synopsis'] = process_item_rating(data_point['item_synopsis'])
            
            s_item = "book"
            l_item = "Book"
        
            input_text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>"
                f"You are a helpful AI assistant for {s_item} recommendation. Based on the user's preferences and {s_item} characteristics provided, generate a recommendation reasoning and predict the user's rating."
                "You must always generate a response in the following format whenever the user provides information:"
                f"Reasoning: [Provide a detailed, single-paragraph reasoning for your prediction, addressing at least three specific points of alignment or misalignment between the user's preferences and the {s_item}'s characteristics.]\n"
                f"Predicted User Rating: [Predict the user's rating as an integer from 1 to 5: 1, 2, 3, 4, 5. 1 indicates the user would strongly dislike the {s_item}, while 5 indicates the user would highly enjoy and recommend it. Consider the average ratings provided for the user and the {s_item} in your prediction.]\n"
                "Note: Do not simply repeat the input text. Generate a new reasoning and rating prediction based on the input provided.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>"
                f"I need a recommendation for this {s_item}. Here's the information:\n"
                "User Preferences:\n"
                "<User Persona>\n"
                f"{data_point['user_persona']}\n\n"
                f"{l_item} Characteristics:\n"
                f"<{l_item} Description>\n"
                f"{data_point['item_description']}\n\n"
                f"<{l_item} Synopsis>\n"
                f"{data_point['item_synopsis']}\n\n"
                f"Based on this information, please provide a detailed reasoning for your recommendation and predict a rating for this {s_item}. Follow the format specified in the system instructions.<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )
        full_prompt = f"{input_text}"
    
        return {"prompt": full_prompt}
    
    

    if test_data_path.endswith(".json"):
        test_data = load_dataset("json", data_files=test_data_path)
    else:
        test_data = load_dataset(test_data_path)
        

    test_data = test_data["train"].map(generate_and_tokenize_prompt)
    
    sampling_params = SamplingParams(max_tokens=512, temperature=0.0, seed=seed)
    
    llm = LLM(model=saved_path, tensor_parallel_size=1, seed=seed, tokenizer=saved_path)
    
    output = llm.generate([item['prompt'] for item in test_data], sampling_params, use_tqdm=True)
    
    new_prompts = []
    for i, item in enumerate(output):
        try:
            reasoning = item.outputs[0].text.split("Predicted User Rating: ")[0].strip().split("Reasoning: ")[1].strip()
        except IndexError:
            print(f"Warning: Reasoning not found for item {i}. Using dummy value.")
            reasoning = "No reasoning provided."
        
        try:
            rating = item.outputs[0].text.split("Predicted User Rating:")[1].split()[0]
        except IndexError:
            print(f"Warning: Rating not found for item {i}. Using dummy value.")
            rating = "-1"  
        
        new_prompt = f"{test_data[i]['prompt']}\nReasoning: {reasoning}\nPredicted User Rating: "
        new_prompts.append(new_prompt)
    
    logprob_sampling_params = SamplingParams(temperature=0.0, max_tokens=1, use_beam_search=False, logprobs=20, seed=seed)  

    new_outputs = llm.generate(new_prompts, logprob_sampling_params, use_tqdm=True)

    results = {}
    for i, (item, new_output) in enumerate(zip(output, new_outputs)):
        try:
            reasoning = item.outputs[0].text.split("Predicted User Rating: ")[0].strip().split("Reasoning: ")[1].strip()
        except IndexError:
            reasoning = "No reasoning provided."
        
        try:
            rating = item.outputs[0].text.split("Predicted User Rating:")[1].split()[0]
        except IndexError:
            rating = "-1"
        
        logprobs = new_output.outputs[0].logprobs[0]
        
        digit_logprobs = {}
        for token, logprob in logprobs.items():
            if is_valid_digit(logprob.decoded_token, dataset):
                digit_logprobs[logprob.decoded_token] = logprob.logprob
        
        if not digit_logprobs:
            print(f"Warning: No valid digit logprobs found for item {i}. Using -1 for all values.")
            expected_rating = -1
            max_prob_rating = "-1"
            probabilities = {str(j): 0.0 for j in (range(1, 11) if dataset == "imdb" else range(1, 6))}
        else:
            logprob_values = np.array(list(digit_logprobs.values()))
            probs = softmax(logprob_values, temp=1.0)
            
            if dataset == "imdb":
                expected_rating = np.sum([(int(k) + 1) * p for k, p in zip(digit_logprobs.keys(), probs)])
                max_prob_rating = str(int(max(digit_logprobs, key=lambda k: digit_logprobs[k])) + 1)
                probabilities = {str(int(k) + 1): p for k, p in zip(digit_logprobs.keys(), probs)}
                probabilities = {str(j): probabilities.get(str(j), 0.0) for j in range(1, 11)}
            else:
                expected_rating = np.sum([int(k) * p for k, p in zip(digit_logprobs.keys(), probs)])
                max_prob_rating = max(digit_logprobs, key=lambda k: digit_logprobs[k])
                probabilities = {k: p for k, p in zip(digit_logprobs.keys(), probs)}
                probabilities = {str(j): probabilities.get(str(j), 0.0) for j in range(1, 6)}

        results[str(i)] = {
            "generated_text": item.outputs[0].text,
            "reasoning": reasoning,
            "max_prob_rating": max_prob_rating,
            "expected_rating": float(expected_rating),
            "probabilities": probabilities,
        }

    with open(output_dir, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Results have been saved to {output_dir}")

if __name__ == "__main__":
    fire.Fire(test)
