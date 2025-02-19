import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainingArguments, TextStreamer, AutoTokenizer, DataCollatorForSeq2Seq
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import FastLanguageModel, is_bfloat16_supported


max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
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
    # mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    chat_template="llama-3.2",
)

# tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")

# print(tokenizer)
from unsloth.chat_templates import standardize_sharegpt
# dataset = dataset.map(formatting_prompts_func, batched = True,)


def apply_template(examples):
    messages = examples['chosen']
    text = []
    for msg in messages:
        prompt_msg = [
            {"role": "system", "content": "You're a and AI Assisstant that helps answering questions"},
        ]
        prompt_msg.extend(msg)
        # text.append(prompt_msg)
        text.append(tokenizer.apply_chat_template(prompt_msg, tokenize=False, add_generation_prompt=False))
        # print(text)
        # print('===')
    return {"text": text}

dataset_name = "mlabonne/orpo-dpo-mix-40k"
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.shuffle(seed=42).select(range(20000))
dataset = dataset.map(apply_template, batched=True)
# dataset = standardize_sharegpt(dataset)

# import sys; sys.exit()
print('training time...')

def formatting_func(example):
    # print(example)
    return example

# sft trainer can detect the conerstional data format and do the pre-processing on its own
dataset_kwargs = {"skip_prepare_dataset": True}
trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer = tokenizer),
    # formatting_func=formatting_func,
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
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        output_dir="output",
        seed=0,
        # dataset_kwargs=dataset_kwargs,
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

trainer.train()

model = FastLanguageModel.for_inference(model)

messages = [
    {"from": "human", "value": "Is 9.11 larger than 9.9?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

text_streamer = TextStreamer(tokenizer)
_ = model.generate(input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True)

model.save_pretrained_merged("model", tokenizer, save_method="merged_16bit")
model.push_to_hub_merged("KickItLikeShika/Llama-3.1-1B-sft-20k", tokenizer, save_method="merged_16bit")

quant_methods = ["q2_k", "q3_k_m", "q4_k_m", "q5_k_m", "q6_k", "q8_0"]
for quant in quant_methods:
    model.push_to_hub_gguf("KickItLikeShika/Llama-3.1-1B-sft-20k-GGUF", tokenizer, quant)