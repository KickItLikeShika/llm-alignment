import gc
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

def train_orpo(train_dataset=None, base_model_path=None):
    """
    Train the model with ORPO.
    Args:
        train_dataset: The dataset to train on.
        base_model_path: Optional path to a fine-tuned model to use as base instead of the default.
    Returns:
        The trained model and tokenizer.
    """
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16 
        
    base_model = base_model_path if base_model_path else "meta-llama/Llama-3.2-1B"
    new_model = "ORPOLlama-3.2-1B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_implementation
    )
    
    # only set up chat format if it doesn't already exist, because it might already be set up in the SFT model
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("using SFT model, setting up chat format...")
        model, tokenizer = setup_chat_format(model, tokenizer)

    model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if train_dataset is None:
        dataset_name = "mlabonne/orpo-dpo-mix-40k"
        dataset = load_dataset(dataset_name, split="all")
        dataset = dataset.shuffle(seed=42).select(range(20000))
    else:
        dataset = train_dataset

    def format_chat_template(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    dataset = dataset.map(
        format_chat_template,
        num_proc=os.cpu_count(),
    )

    # just to evalaute while training
    dataset = dataset.train_test_split(test_size=0.01)

    print("\nExample formatted training data:")
    for i in range(2):
        print(f"\nExample {i+1}:")
        print("Chosen text:")
        print(dataset["train"][i]["chosen"])
        print("\nRejected text:")
        print(dataset["train"][i]["rejected"])
        print("-" * 80)

    orpo_args = ORPOConfig(
        learning_rate=8e-6,
        beta=0.1,
        lr_scheduler_type="linear",
        max_length=2048,
        max_prompt_length=512,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=0.2,
        logging_steps=1000,
        warmup_steps=10,
        report_to="wandb",
        output_dir="./results/",
    )

    trainer = ORPOTrainer(
        model=model,
        args=orpo_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(new_model)

    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        print("using ORPO model, setting up chat format...")
        model, tokenizer = setup_chat_format(model, tokenizer)

    model = PeftModel.from_pretrained(model, new_model)
    model = model.merge_and_unload()

    model.push_to_hub(new_model, use_temp_dir=False)
    tokenizer.push_to_hub(new_model, use_temp_dir=False)

    return model, tokenizer
