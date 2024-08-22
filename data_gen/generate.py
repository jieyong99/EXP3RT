import argparse
import asyncio
import json
import os
import random
from copy import deepcopy
import pandas as pd
import random

from tqdm import tqdm
from nltk import word_tokenize
from tqdm.asyncio import tqdm_asyncio

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

TOTAL_COST = 0  # making this a global variable, be aware this may lead to issues in concurrent scenarios
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_owner", choices=["your_api_key"], default="your_api_key", help="Make sure you have config/profile_info.json and both org and api info in it.")
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--task", choices=["generate_preference", "generate_user_side", "generate_item_side", "generate_rationale"], type=str, required=True, help="Which task to generate data for.")
    parser.add_argument("--prompt_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, required=True, help="It should be a NEW DIRECTORY. Please do not use an existing one.")
    parser.add_argument("--num_sample", type=int, default=None, help="If you want to test your code by sampling a small number of data, you can set this argument.")
    parser.add_argument("--model_name", choices=["gpt-3.5-turbo-1106", "gpt-4-turbo"], type=str, default="gpt-3.5-turbo-1106", help="OpenAI model name. Options: gpt-3.5-turbo-1106 or gpt-4-turbo")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=200)
    args = parser.parse_args()
    if args.num_sample:
        args.save_dir = args.save_dir + f"_sample{args.num_sample}"
    return args

def set_openai_api(api_owner):
    profilel_info = json.load(open("config/profilel_info.json", "r"))
    os.environ["OPENAI_API_KEY"] = profilel_info[api_owner]["api_key"]
    os.environ["OPENAI_ORGANIZATION"] = profilel_info[api_owner]["org_id"]
    print(f"Set OpenAI API Key and Organization of {api_owner}.")

## TODO : change this function to load the prompt from your own file ##
def load_prompt(args):
    """
    Load .txt file as a prompt.
    """
    if args.prompt_path:
        with open(args.prompt_path, 'r') as f:
            prompt = f.read()
    return prompt


def prepare_model_input_preference(prompt:str, data_path:str):
    '''
        input : prompt, data_path (str)
        output : all_model_data (list of dict)
    '''
    print("Loading data for generating preferences from reviews...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    all_model_data = []
    for d in tqdm(data):
        input_temp = dict()
        input_temp["user_id"] = d["user_id"]
        input_temp["item_id"] = d["item_id"]
        input_temp["review"] = d["review"]
        input_temp['model_input'] = prompt.format(**{
                    "review": d['review']
                })
        all_model_data.append(input_temp)

    return all_model_data

def prepare_model_input_item(prompt:str, data_path:str):
    '''
        input : prompt, data_path (str)
        output : all_model_data (list of dict)
    '''
    print("Loading data for generating item side preferences...")
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    all_model_data = []
    for d in tqdm(data):
        input_temp = dict()
        input_temp["item_id"] = d["item_id"]
        input_temp["preference_set"] = d["preference_set"]
        input_temp['model_input'] = prompt.format(**{
                    "preferences": d['preference_set']
                })
        all_model_data.append(input_temp)

    return all_model_data


def prepare_model_input_user(prompt:str, data_path:str):
    '''
        input : prompt, data_path (str)
        output : all_model_data (list of dict)
    '''
    print("Loading data for generating user side profile...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    all_model_data = []
    for d in tqdm(data):
        input_temp = dict()
        input_temp["user_id"] = d["user_id"]
        input_temp["preference_set"] = d["preference_set"]
        input_temp['model_input'] = prompt.format(**{
                    "preferences": d['preference_set']
                })
        all_model_data.append(input_temp)

    return all_model_data

def prepare_model_input_rationale(prompt:str, data_path:str):
    '''
        input : prompt, data_path (str)
        output : all_model_data (list of dict)
    '''
    print("Loading data for generating rationale...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    all_model_data = []
    for d in tqdm(data):
        input_temp = dict()
        input_temp["user_id"] = d["user_id"]
        input_temp["item_id"] = d["item_id"]
        input_temp["user_rating"] = d["user_rating"]
        input_temp["user_profile"] = d["user_profile"]
        input_temp["item_description"] = d["item_description"]
        input_temp["item_profile"] = d["item_profile"]
        input_temp["instruction"] = "You are a recommender who provides a reason for whether or not to recommend a specific item to a user. \nYou have have extensive information about the user and item : <User Persona>, <Item Description> and <Item Synopsis>.\nBased on this preferences, create a explanation of whether or not to recommend it. \n\nLet\'s think step by step. \n1. Read <User Persona> and think about the user\'s preference. \n2. Connect the user\'s preference with item description and item knowledge, in which specific point the user will like/dislike this item. \n3. Create explanation of recommendation based on the given information. \n4. Rate how much the user will like it on a scale of 5.0."

        input_temp['model_input'] = prompt.format(**{
            "user_rating": d["user_rating"],
            "user_profile": d["user_profile"],
            "item_description": d["item_description"],
            "item_profile" : d["item_profile"]

        })

        all_model_data.append(input_temp)
    return all_model_data

def load_and_prepare_data(args):
    prompt = load_prompt(args)
    print("Preparing model inputs...")

    if args.task == "generate_preference":
        all_model_data = prepare_model_input_preference(
            prompt, args.input_path)
    elif args.task == "generate_user_side":
        all_model_data = prepare_model_input_user(
            prompt, args.input_path)
    elif args.task == "generate_item_side":
        all_model_data = prepare_model_input_item(
            prompt, args.input_path)
    elif args.task == "generate_rationale":
        all_model_data = prepare_model_input_rationale(
            prompt, args.input_path)
    return all_model_data


def sample_indices(all_model_inputs, num_sample):
    random.seed(0)
    cand_indices = list(range(len(all_model_inputs)))
    sampled_indices = random.sample(cand_indices, num_sample)
    return sampled_indices


def filter_data(all_model_data, num_sample):
    if num_sample:
        sampled_indices = sample_indices(all_model_data, num_sample)
        all_model_data = [all_model_data[i] for i in sampled_indices]
    return all_model_data


async def async_generate(llm, model_data, idx, save_dir):
    global TOTAL_COST
    system_message = SystemMessage(content=model_data['model_input'])
    # human_message = HumanMessage(content=model_data['model_input']) # if you need it
    while True:
        try:
            response = await llm.agenerate([[system_message]])
            # response = await llm.agenerate([[system_message, human_message]]) # if you need it
            input_tokens = response.llm_output['token_usage']['prompt_tokens']
            output_tokens = response.llm_output['token_usage']['completion_tokens']
            if llm.model_name == "gpt-3.5-turbo-1106":
                TOTAL_COST += input_tokens / 1000 * 0.001
                TOTAL_COST += output_tokens / 1000 * 0.002
            elif llm.model_name == "gpt-4-1106-preview":
                TOTAL_COST += input_tokens / 1000 * 0.01
                TOTAL_COST += output_tokens / 1000 * 0.03
            print(idx, TOTAL_COST)
            break
        except Exception as e:
            print(f"Exception occurred: {e}")
            response = None

    result = deepcopy(model_data) ## TODO : change this code if you want to save it in a different way
    result['output'] = response.generations[0][0].text
    with open(os.path.join(save_dir, f"{idx}.json"), "w", encoding='UTF-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    return result


async def generate_concurrently(all_model_data, start_idx, args):
    llm = ChatOpenAI(model_name=args.model_name,
                     temperature=args.temperature,
                     max_tokens=args.max_tokens,
                     max_retries=100)
    
    tasks = [async_generate(llm, model_data, i+start_idx, args.save_dir)
             for i, model_data in enumerate(all_model_data)]
    return await tqdm_asyncio.gather(*tasks)


async def main(args):
    all_model_data = load_and_prepare_data(args)
    all_model_data = filter_data(all_model_data, args.num_sample)

    args.save_dir = args.save_dir + f"_temp{args.temperature}"
    # if exist, make a new dir
    while os.path.exists(args.save_dir):
        args.save_dir = args.save_dir + "_new"
    os.makedirs(args.save_dir, exist_ok=False)

    all_results = []
    if len(all_model_data) > 1000:
        for start_idx in tqdm(range(0, len(all_model_data), 1000)):
            cur_model_data = all_model_data[start_idx:start_idx+1000]
            all_results.extend(await generate_concurrently(cur_model_data, start_idx, args))
    else:
        all_results = await generate_concurrently(all_model_data, 0, args)

    total_result_path = args.save_dir + "_total_results.json"
    with open(os.path.join(total_result_path), "w", encoding='UTF-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
        
if __name__ == "__main__":
    args = parse_args()
    set_openai_api(args.api_owner)

    import json
    # train/valid/test ids
    with open('./ids/train_ids.json', 'r') as f:
        train_ids = json.load(f)
    with open('./ids/valid_ids.json', 'r') as f:
        valid_ids = json.load(f)
    with open('./ids/test_ids.json', 'r') as f:
        test_ids = json.load(f)

    # user/item ids
    with open('./data/book/ids/user_ids.json','r') as f:
        user_list = json.load(f)
    with open('./data/book/ids/item_ids.json','r') as f:
        item_list = json.load(f)

    asyncio.run(main(args))
