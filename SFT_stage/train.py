import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

def parse_args():
    parser = argparse.ArgumentParser(description="SFT Training Script for Qwen/LLMs")
    
    # Model and Dataset paths
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="Path or HF ID of the base model")
    parser.add_argument("--dataset_name", type=str, default="HINT-lab/sft_Qwen_Qwen3-0.6B", help="Path or HF ID of the dataset")
    parser.add_argument("--output_dir", type=str, default="./results_0.6B", help="Directory to save results")
    
    # Training Hyperparameters
    parser.add_argument("--max_seq_length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Warmup ratio")
    
    # Optimization flags
    parser.add_argument("--use_gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save VRAM")
    # Note: Setting default to True below to match your original script's intent
    parser.set_defaults(use_gradient_checkpointing=True)
    
    # Template specific
    parser.add_argument("--response_template", type=str, default="<|im_start|>assistant\n", help="The template string indicating the start of the assistant response")

    return parser.parse_args()

def main():
    args = parse_args()

    # 0. Avoid Hugging Face Tokenizer parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"--- Configuration ---")
    print(f"Model: {args.model_name}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Max Seq Len: {args.max_seq_length}")
    print(f"---------------------")

    # ---------- 2. Load Model and Tokenizer ----------
    print("Loading model and tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            enable_thinking=False
        )
    except TypeError:
        print("Warning: 'enable_thinking' arg not supported by this tokenizer, removing it.")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )

    # --- Critical Fix 1: Set pad_token ---
    if tokenizer.pad_token is None:
        print("Setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- 3. Load and Format Dataset ----------
    print("Loading and formatting dataset...")
    dataset = load_dataset(args.dataset_name, split="train")

    # --- Critical Fix 2: Formatting ---
    def format_example(example):
        input_text = example["question"] + "\n" + r"Please reason step by step, and put your final answer within \\boxed{}."
        return {
            "messages": [
                {"role": "user", "content": input_text},
                {"role": "assistant", "content": example["answer"]}
            ]
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # --- Critical Fix 3: Filter empty answers ---
    dataset = dataset.filter(
        lambda example: example["messages"][1]["content"] is not None and len(example["messages"][1]["content"].strip()) > 0
    )

    # --- Critical Fix 4: Manually apply chat template (to work with DataCollator) ---
    def apply_chat_template(example):
        formatted_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False 
        )
        return {"text": formatted_text}

    dataset = dataset.map(apply_chat_template, remove_columns=["messages"])

    print(f"Sample Text: {dataset[0]['text']}")

    # ---------- 4. (New) Set Data Collator to mask User Loss ----------
    # Qwen templates usually have "<|im_start|>assistant\n" before the assistant response.
    # We pass this via command line args.
    response_template = args.response_template

    # Note: If training loss is consistently 0, check if this template matches the tokenization exactly.
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, 
        tokenizer=tokenizer
    )

    # ---------- 5. Trainer Configuration ----------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length, 
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=10,
        gradient_checkpointing=args.use_gradient_checkpointing, # Strongly recommended for VRAM saving
        save_strategy="epoch",       # Avoid saving too many checkpoints
        report_to="none"             # Avoid hanging if wandb is not configured
    )

    # ---------- 6. Create Trainer ----------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator, 
    )

    # ---------- 7. Start Training ----------
    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()