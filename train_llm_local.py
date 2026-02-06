#!/usr/bin/env python3
"""
Train Chess LLM Locally

Fine-tune LLMs to predict chess position evaluations.
Requires GPU with 8GB+ VRAM.

Usage:
    python3 train_llm_local.py --data chess_llm_alpaca_50000_train.jsonl
    python3 train_llm_local.py --data chess_llm_alpaca_50000_train.jsonl --model llama
"""

import argparse
import json
import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: No GPU detected. Training will be very slow.")
            print("Consider using Google Colab with train_chess_llm_colab.ipynb")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(0)
    except ImportError:
        print("ERROR: PyTorch not installed. Run: pip install torch")
        sys.exit(1)

    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("ERROR: Unsloth not installed. Run:")
        print("  pip install unsloth")
        print("  pip install --no-deps trl peft accelerate bitsandbytes")
        sys.exit(1)

def load_data(data_path: str):
    """Load JSONL training data."""
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            # Convert alpaca format to text format
            if 'instruction' in entry:
                text = f"""### Instruction:
{entry['instruction']}

### Input:
{entry['input']}

### Response:
{entry['output']}"""
                data.append({"text": text})
            elif 'text' in entry:
                data.append(entry)
            elif 'conversations' in entry:
                # ShareGPT format
                conv = entry['conversations']
                text = f"### Human:\n{conv[0]['value']}\n\n### Assistant:\n{conv[1]['value']}"
                data.append({"text": text})
    return data

MODELS = {
    "phi": "unsloth/Phi-3.5-mini-instruct",      # 3.8B - fast, good quality
    "llama": "unsloth/Llama-3.2-3B-Instruct",    # 3B - good balance
    "qwen": "unsloth/Qwen2.5-3B-Instruct",       # 3B - multilingual
    "mistral": "unsloth/mistral-7b-instruct-v0.3",  # 7B - needs more VRAM
}

def train(args):
    """Run training."""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Load model
    model_name = MODELS.get(args.model, args.model)
    print(f"Loading model: {model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print(f"Trainable parameters: {model.print_trainable_parameters()}")

    # Load data
    print(f"Loading data from: {args.data}")
    train_data = load_data(args.data)
    print(f"Loaded {len(train_data):,} training samples")

    # Load validation data if exists
    val_data = None
    val_path = args.data.replace("_train.jsonl", "_val.jsonl")
    if Path(val_path).exists():
        val_data = load_data(val_path)
        print(f"Loaded {len(val_data):,} validation samples")

    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data) if val_data else None

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=25,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        save_strategy="steps",
        save_steps=200,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=training_args,
    )

    # Train
    print("\nStarting training...")
    stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"Total time: {stats.metrics['train_runtime'] / 60:.1f} minutes")

    # Save
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")

    return model, tokenizer

def test_model(model, tokenizer):
    """Test the trained model."""
    from unsloth import FastLanguageModel

    FastLanguageModel.for_inference(model)

    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Italian Game"),
    ]

    print("\n" + "=" * 60)
    print("Testing model predictions:")
    print("=" * 60)

    for fen, description in test_positions:
        prompt = f"""### Instruction:
Evaluate this chess position.

### Input:
{fen}

### Response:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("### Response:")[-1].strip()

        print(f"\nPosition: {description}")
        print(f"FEN: {fen[:50]}...")
        print(f"Prediction: {response[:150]}")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Train Chess LLM locally")

    parser.add_argument("--data", "-d", type=str, required=True,
                        help="Path to training data (JSONL)")
    parser.add_argument("--model", "-m", type=str, default="phi",
                        choices=list(MODELS.keys()),
                        help="Model to fine-tune (default: phi)")
    parser.add_argument("--output-dir", "-o", type=str, default="./chess_llm_output",
                        help="Output directory for model")
    parser.add_argument("--epochs", "-e", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max-seq-length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip testing after training")

    args = parser.parse_args()

    check_dependencies()
    model, tokenizer = train(args)

    if not args.skip_test:
        test_model(model, tokenizer)

    print("\nDone!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
