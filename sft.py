import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer, AutoTokenizer, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import FastLanguageModel, is_bfloat16_supported

def train_sft(train_dataset=None):
    """
    Train the model with supervised finetuning.
    Args:
        train_dataset: The dataset to train on.
    Returns:
        The trained model and tokenizer.
    """
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-1B",
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        dtype=torch.bfloat16,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        lora_alpha=16,
        lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
        use_rslora=True,
        use_gradient_checkpointing="unsloth"
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.2",
    )

    def apply_template(examples):
        messages = examples['chosen']
        text = []
        for msg in messages:
            prompt_msg = [
                {"role": "system", "content": "You're a and AI Assisstant that helps answering questions"},
            ]
            prompt_msg.extend(msg)
            text.append(tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=False))
        return {"text": text}

    if train_dataset is None:
        dataset_name = "mlabonne/orpo-dpo-mix-40k"
        train_dataset = load_dataset(dataset_name, split="train")
        train_dataset = train_dataset.shuffle(seed=42).select(range(20000))
    
    train_dataset = train_dataset.map(apply_template, batched=True)

    print("\nExample formatted training data:")
    for i in range(2):
        print(f"\nExample {i+1}:")
        print(train_dataset[i]["text"])
        print("-" * 80)

    print('training time...')

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        args=SFTConfig(
            learning_rate=3e-4,
            packing=False,
            dataset_num_proc=4,
            max_seq_length=max_seq_length,
            dataset_text_field="text",
            lr_scheduler_type="linear", 
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            num_train_epochs=1,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1000,
            optim="adamw_8bit",
            weight_decay=0.01,
            warmup_steps=10,
            output_dir="output",
            seed=0,
        ),
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    trainer.train()

    model = FastLanguageModel.for_inference(model)

    model.save_pretrained_merged("results/sft", tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged("KickItLikeShika/SFTLlama-3.2-1B", tokenizer, save_method="merged_16bit")

    return model, tokenizer