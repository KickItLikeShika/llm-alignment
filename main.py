import os
import gc
import torch
import wandb
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import setup_chat_format
import evaluate
from sft import train_sft
from orpo import train_orpo

DATASET_NAME = "mlabonne/orpo-dpo-mix-40k"
DATASET_PATH = "./data"
BASE_MODEL = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_and_prepare_dataset():
    """
    Load and prepare the dataset.
    If the dataset is already cached, load it from the cache.
    Otherwise, download and prepare the dataset.
    """
    if os.path.exists(f"{DATASET_PATH}/train.json") and os.path.exists(f"{DATASET_PATH}/val.json"):
        print("Loading cached dataset...")
        with open(f"{DATASET_PATH}/train.json", 'r') as f:
            train_dataset = load_dataset('json', data_files=f"{DATASET_PATH}/train.json")['train']
        with open(f"{DATASET_PATH}/val.json", 'r') as f:
            val_dataset = load_dataset('json', data_files=f"{DATASET_PATH}/val.json")['train']
    else:
        print("Downloading and preparing dataset...")
        dataset = load_dataset(DATASET_NAME, split="all", token=True)
        dataset = dataset.shuffle(seed=42).select(range(20000))

        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset["train"]
        val_dataset = split_dataset["test"]

        os.makedirs(DATASET_PATH, exist_ok=True)
        train_dataset.to_json(f"{DATASET_PATH}/train.json")
        val_dataset.to_json(f"{DATASET_PATH}/val.json")

    return train_dataset, val_dataset

def evaluate_model(model, tokenizer, dataset, split_name="validation"):
    """
    Evaluate the model on the given dataset.
    Only evaluate the first 1000 examples.
    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer to use.
        dataset: The dataset to evaluate on.
        split_name: The name of the split to evaluate on.
    Returns:
        The average ROUGE scores.
    """
    print(f"\nEvaluating model on {split_name} set...")
    model.eval()

    rouge = evaluate.load('rouge')
    total_rouge_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0}

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print(f"Evaluated {i} examples...")
            
        # only evaluate first 1000 examples
        if i > 1000:
            break

        # extract the assistant's response from the chosen messages
        chosen_messages = example['chosen']
        assistant_message = next(msg['content'] for msg in chosen_messages if msg['role'] == 'assistant')
        user_message = next(msg['content'] for msg in chosen_messages if msg['role'] == 'user')

        # generate response to user message
        inputs = tokenizer(user_message, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference_text = assistant_message

        scores = rouge.compute(predictions=[generated_text], references=[reference_text])
        for metric in total_rouge_scores:
            total_rouge_scores[metric] += scores[metric]

    # calc average scores
    num_examples = len(dataset)
    avg_scores = {metric: score/num_examples for metric, score in total_rouge_scores.items()}
    print(f"Average ROUGE scores: {avg_scores}")
    return avg_scores

def main():
    """
    Main function to train and evaluate the model.
    """
    train_dataset, val_dataset = load_and_prepare_dataset()
    
    print("\loading base model for initial eval...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=True
    )
    model, tokenizer = setup_chat_format(model, tokenizer)
    
    print("\nperforming initial eval...")
    initial_scores = evaluate_model(model, tokenizer, val_dataset)
    print(f"Base Model: {initial_scores}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # train with SFT
    print("\training with SFT...")
    model, tokenizer = train_sft(train_dataset)
    
    # eval after SFT
    print("\eval SFT model...")
    sft_scores = evaluate_model(model, tokenizer, val_dataset)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # train with ORPO
    print("\nTraining with ORPO...")
    model, tokenizer = train_orpo(train_dataset)

    # eval after ORPO
    print("\nEvaluating ORPO model...")
    orpo_scores = evaluate_model(model, tokenizer, val_dataset)

    print("\n\nResults:")
    print(f"Base Model: {initial_scores}")
    print(f"After SFT: {sft_scores}")
    print(f"After ORPO: {orpo_scores}")

if __name__ == "__main__":
    main()

# Base Model: {'rouge1': 0.14786865275767497, 'rouge2': 0.05741813089556658, 'rougeL': 0.09560573415051726}
# SFT Model: {'rouge1': 0.16640010568749722, 'rouge2': 0.06878433704191969, 'rougeL': 0.1018761352773121}
# ORPO Model: {'rouge1': 0.1505326146060846, 'rouge2': 0.05853340860919248, 'rougeL': 0.09578936523988472}
