# fine_tune_gpt2.py

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
import torch
from functools import partial


MODEL_NAME = "openai-community/gpt2"
OUTPUT_DIR = "models/gpt2_squad"   # folder to save finetuned model


def format_qa(example):
    """
    Turn a Q/A pair into a single text string the model will learn to generate.
    We enforce our fixed style here.
    """
    question = example["question"]
    # SQuAD's "answers" is a dict with "text" list
    answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
    return (
        f"Question: {question}\n"
        f"Answer: That is a great question. {answer} Let me know if you have any other questions."
    )


def main():
    # 1) Load dataset
    squad = load_dataset("squad")

    # Small subset for faster training
    train_ds = squad["train"].select(range(1000))
    valid_ds = squad["validation"].select(range(200))

    # 2) Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 has no pad token by default → add one
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.resize_token_embeddings(len(tokenizer))

    # 3) Map Q/A to text + tokenize
    def preprocess(example, tokenizer):
        text = format_qa(example)
        out = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        out["labels"] = out["input_ids"].copy()
        return out

    preprocess_fn = partial(preprocess, tokenizer=tokenizer)

    tokenized_train = train_ds.map(preprocess_fn, batched=False)
    tokenized_valid = valid_ds.map(preprocess_fn, batched=False)

    # 4) Training arguments
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],   # disable wandb etc.
    )

    # 5) Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 6) Save model + tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()