import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import sys
import fire
import torch
import transformers
from datasets import load_dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
import re
import random
import numpy as np
##
import json
import sys
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
    BitsAndBytesConfig,
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
            def imdb_process_user_rating(text):
                parts = text.split("[Average Rating]")
                if len(parts) > 1:
                    try:
                        rating = float(parts[-1].strip())
                        new_rating = max(0, rating - 1)  # 1을 빼고 0 이상으로 유지
                        return f"{parts[0]}[User Average Rating]\n{new_rating:.1f}"
                    except ValueError:
                        return text
                return text
            def imdb_process_item_rating(text):
                parts = text.split("[Average Rating]")
                if len(parts) > 1:
                    try:
                        rating = float(parts[-1].strip())
                        new_rating = max(0, rating - 1)  # 1을 빼고 0 이상으로 유지
                        return f"{parts[0]}[Item Average Rating]\n{new_rating:.1f}"
                    except ValueError:
                        return text
                return text

            # User Persona와 Item Synopsis의 평균 레이팅 처리
            data_point['user_persona'] = imdb_process_user_rating(data_point['user_persona'])
            data_point['item_synopsis'] = imdb_process_item_rating(data_point['item_synopsis'])
            
            s_item = "movie"
            l_item = "Movie"
            
            input_text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"You are a helpful AI assistant for {s_item} recommendation. Based on the user's preferences and {s_item} characteristics provided, generate a recommendation reasoning and predict the user's rating.\n"
                "You must always generate a response in the following format whenever the user provides information:\n"
                f"Reasoning: [Provide a detailed, single-paragraph reasoning for your prediction, addressing at least three specific points of alignment or misalignment between the user's preferences and the {s_item}'s characteristics.]\n"
                f"Predicted User Rating: [Predict the user's rating as an integer from 0 to 9: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. 0 indicates the user would strongly dislike the {s_item}, while 9 indicates the user would highly enjoy and recommend it. Consider the average ratings provided for the user and the {s_item} in your prediction.]\n"
                "Note: Do not simply repeat the input text. Generate a new reasoning and rating prediction based on the input provided.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
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

            reasoning = data_point['rationale']
            rating = data_point['score']

            output_text = (
                f"Reasoning: {reasoning}\n"
                f"Predicted User Rating: {int(rating)-1}<|eot_id|><|end_of_text|>"
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

            # User Persona와 Item Synopsis의 평균 레이팅 처리
            data_point['user_persona'] = process_user_rating(data_point['user_persona'])
            data_point['item_synopsis'] = process_item_rating(data_point['item_synopsis'])
            
            s_item = "book"
            l_item = "Book"
        
            input_text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
                f"You are a helpful AI assistant for {s_item} recommendation. Based on the user's preferences and {s_item} characteristics provided, generate a recommendation reasoning and predict the user's rating."
                "You must always generate a response in the following format whenever the user provides information:"
                f"Reasoning: [Provide a detailed, single-paragraph reasoning for your prediction, addressing at least three specific points of alignment or misalignment between the user's preferences and the {s_item}'s characteristics.]\n"
                f"Predicted User Rating: [Predict the user's rating as an integer from 1 to 5: 1, 2, 3, 4, 5. 1 indicates the user would strongly dislike the {s_item}, while 5 indicates the user would highly enjoy and recommend it. Consider the average ratings provided for the user and the {s_item} in your prediction.]\n"
                "Note: Do not simply repeat the input text. Generate a new reasoning and rating prediction based on the input provided.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n"
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

            reasoning = data_point['rationale']
            rating = data_point['score']

            output_text = (
                f"Reasoning: {reasoning}\n"
                f"Predicted User Rating: {rating}<|eot_id|><|end_of_text|>"
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
 
        
    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        def extract_rating(text):
            # 'Predicted User Rating:' 문구가 있는지 확인
            if 'Predicted User Rating:' not in text:
                return None
            
            # 'Predicted User Rating:' 이후의 텍스트만 고려
            rating_text = text.split('Predicted User Rating:')[-1].strip()
            
            # 정규 표현식을 사용하여 첫 번째로 나오는 한 자리 정수를 찾음
            match = re.search(r'\b(\d)\b', rating_text)
            
            if match:
                return int(match.group(1))
            
            return None

        pred_ratings = []
        true_ratings = []
        sample_comparisons = []
        total_predictions = 0
        total_true_ratings = 0
    
        batch_size = micro_batch_size
        for i in range(0, len(predictions), batch_size):
            batch_preds = predictions[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            with torch.no_grad():
                valid_preds = [[token for token in seq if 0 <= token < len(tokenizer)] for seq in batch_preds]
                valid_labels = [[token for token in seq if 0 <= token < len(tokenizer) and token != -100] for seq in batch_labels]
                
                try:
                    pred_texts = tokenizer.batch_decode(valid_preds, skip_special_tokens=True)
                    label_texts = tokenizer.batch_decode(valid_labels, skip_special_tokens=True)
                    
                except IndexError as e:
                    print(f"Error in batch {i}: {e}")
                    continue
            
            
            for pred_text, label_text in zip(pred_texts, label_texts):
                total_predictions += 1
                pred_rating = extract_rating(pred_text)
                true_rating = extract_rating(label_text)
                
                if true_rating is not None:
                    total_true_ratings += 1
                    
                    if pred_rating is not None:
                        pred_ratings.append(pred_rating)
                        true_ratings.append(true_rating)
                        
                        if random.random() < 0.1:
                            sample_comparisons.append((true_rating, pred_rating, pred_text))
            
            torch.cuda.empty_cache()

        if len(pred_ratings) == 0:
            print("No valid ratings extracted!")
            return {
                "rmse": float('inf'),
                "mae": float('inf'),
                "accuracy": 0.0,
                "valid_ratio": 0.0
            }

        # 샘플 비교 출력 (최대 10개)
        print("\nSample Rating Comparisons:")
        for true, pred, text in sample_comparisons[:10]:  # 여기를 3개 값을 언패킹하도록 수정
            print(f"TRUE: {true}  PRED: {pred}")

        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        mae = mean_absolute_error(true_ratings, pred_ratings)
        accuracy = sum(p == t for p, t in zip(pred_ratings, true_ratings)) / len(pred_ratings)
        # valid_ratio = len(pred_ratings) / valid_ratings
        valid_ratio = len(pred_ratings) / total_true_ratings if total_true_ratings > 0 else 0
        
        print(f"\nTotal predictions: {total_predictions}")
        print(f"Total true ratings: {total_true_ratings}")
        print(f"Valid predictions (both true and pred ratings available): {len(pred_ratings)}")


        return {
            "rmse": rmse,
            "mae": mae,
            "accuracy": accuracy,
            "valid_ratio": valid_ratio
        }

    # class SaveBestModelCallback(TrainerCallback):
    #     def __init__(self, save_dir, output_dir, model, tokenizer, rmse_patience=5, valid_ratio_patience=5):
    #         self.save_dir = save_dir
    #         self.output_dir = output_dir
    #         self.model = model
    #         self.tokenizer = tokenizer
    #         self.rmse_patience = rmse_patience
    #         self.valid_ratio_patience = valid_ratio_patience
    #         # self.max_checkpoints = max_checkpoints
    #         self.best_rmse = float('inf')
    #         self.best_valid_ratio = 0.0
    #         self.rmse_no_improvement_count = 0
    #         self.valid_ratio_no_improvement_count = 0
    #         self.checkpoints = []  # 이제 (rmse, epoch, checkpoint_dir) 튜플의 최소 힙으로 사용됩니다

    #     def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
    #         if 'eval_rmse' not in metrics or 'eval_valid_ratio' not in metrics:
    #             return control
            
    #         current_rmse = metrics['eval_rmse']
    #         current_valid_ratio = metrics['eval_valid_ratio']
            
    #         print(f"\nCurrent RMSE: {current_rmse}, Valid ratio: {current_valid_ratio:.2f}")

    #         improved = False

    #         # Check RMSE improvement
    #         if current_rmse < self.best_rmse:
    #             self.best_rmse = current_rmse
    #             self.rmse_no_improvement_count = 0
    #             improved = True
    #         else:
    #             self.rmse_no_improvement_count += 1

    #         # Check Valid Ratio improvement
    #         if current_valid_ratio > self.best_valid_ratio:
    #             self.best_valid_ratio = current_valid_ratio
    #             self.valid_ratio_no_improvement_count = 0
    #             improved = True
    #         else:
    #             self.valid_ratio_no_improvement_count += 1

    #         # 새로운 체크포인트 폴더 생성
    #         checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-epoch{int(state.epoch)}")
    #         os.makedirs(checkpoint_dir, exist_ok=True)
            
    #         # 모델(어댑터) 저장
    #         self.model.save_pretrained(checkpoint_dir)
    #         # 토크나이저 저장
    #         self.tokenizer.save_pretrained(checkpoint_dir)
            
    #         print(f"Model saved with RMSE: {current_rmse} and Valid Ratio: {current_valid_ratio:.2f} at {checkpoint_dir}")
            
    #         # 변경 후
    #         self.checkpoints.append((current_rmse, state.epoch, checkpoint_dir))
                
    #         # 만약 현재 모델이 지금까지 중 가장 좋은 RMSE를 가진다면 output_dir에 저장
    #         if current_rmse == self.best_rmse:
    #             self.model.save_pretrained(self.output_dir)
    #             self.tokenizer.save_pretrained(self.output_dir)
    #             print(f"New best model saved to {self.output_dir}")

    #         if improved:
    #             print(f"New best model! Best RMSE: {self.best_rmse}, Best Valid Ratio: {self.best_valid_ratio:.2f}")
    #         else:
    #             print(f"No improvement. Best RMSE: {self.best_rmse}, Best Valid Ratio: {self.best_valid_ratio:.2f}")

    #         # Check for early stopping
    #         if (self.rmse_no_improvement_count >= self.rmse_patience and 
    #             self.valid_ratio_no_improvement_count >= self.valid_ratio_patience):
    #             print(f"\nEarly stopping triggered. Best RMSE: {self.best_rmse}, Best Valid Ratio: {self.best_valid_ratio:.2f}")
    #             control.should_training_stop = True

    #         return control

    #     def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #         print(f"\nTraining ended. Best RMSE: {self.best_rmse}, Best Valid Ratio: {self.best_valid_ratio:.2f}")
    #         # 변경 후
    #         print("All checkpoints are saved in:")
    #         for _, _, checkpoint in sorted(self.checkpoints):
    #             print(checkpoint)
    #         print(f"Best model is saved in: {self.output_dir}")

    # save_dir = os.path.join(output_dir, "checkpoints")
    # os.makedirs(save_dir, exist_ok=True)
    # custom_callback = SaveBestModelCallback(save_dir=save_dir, output_dir=output_dir, model=model, tokenizer=tokenizer, rmse_patience=rmse_patience, valid_ratio_patience=5)
    
    class SaveBestModelCallback(TrainerCallback):
        def __init__(self, save_dir, output_dir, model, tokenizer, patience=5):
            self.save_dir = save_dir
            self.output_dir = output_dir
            self.model = model
            self.tokenizer = tokenizer
            self.patience = patience
            self.best_loss = float('inf')
            self.loss_no_improvement_count = 0
            self.checkpoints = []

        def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, metrics, **kwargs):
            if 'eval_loss' not in metrics:
                return control
            
            current_loss = metrics['eval_loss']
            
            print(f"\nCurrent Eval Loss: {current_loss}")

            improved = False

            # Check Loss improvement
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.loss_no_improvement_count = 0
                improved = True
            else:
                self.loss_no_improvement_count += 1

            # 새로운 체크포인트 폴더 생성
            checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-epoch{int(state.epoch)}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 모델(어댑터) 저장
            self.model.save_pretrained(checkpoint_dir)
            # 토크나이저 저장
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            print(f"Model saved with Eval Loss: {current_loss} at {checkpoint_dir}")
            
            self.checkpoints.append((current_loss, state.epoch, checkpoint_dir))
                
            # 만약 현재 모델이 지금까지 중 가장 좋은 Loss를 가진다면 output_dir에 저장
            if current_loss == self.best_loss:
                self.model.save_pretrained(self.output_dir)
                self.tokenizer.save_pretrained(self.output_dir)
                print(f"New best model saved to {self.output_dir}")

            if improved:
                print(f"New best model! Best Eval Loss: {self.best_loss}")
            else:
                print(f"No improvement. Best Eval Loss: {self.best_loss}")

            # Check for early stopping
            if self.loss_no_improvement_count >= self.patience:
                print(f"\nEarly stopping triggered. Best Eval Loss: {self.best_loss}")
                control.should_training_stop = True

            return control

        def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
            print(f"\nTraining ended. Best Eval Loss: {self.best_loss}")
            print("All checkpoints are saved in:")
            for _, _, checkpoint in sorted(self.checkpoints):
                print(checkpoint)
            print(f"Best model is saved in: {self.output_dir}")

    # 콜백 인스턴스 생성
    save_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    custom_callback = SaveBestModelCallback(save_dir=save_dir, output_dir=output_dir, model=model, tokenizer=tokenizer, patience=5)
    
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
            # eval_steps=100,
            save_strategy="no",
            output_dir=output_dir,
            ddp_find_unused_parameters=False,
            group_by_length=group_by_length,
            gradient_checkpointing=True,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
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