import os
import argparse
import yaml
import pandas as pd
import numpy as np
import torch
from evaluate import load
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import wandb

def load_imdb_data(data_dir):
    """
    Đọc 2 file CSV train.csv và test.csv từ data_dir,
    chuyển thành HuggingFace Dataset.
    """
    train_path = os.path.join(data_dir, "train.csv")
    test_path  = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_ds = Dataset.from_pandas(train_df)
    test_ds  = Dataset.from_pandas(test_df)
    return train_ds, test_ds

def tokenize_fn(examples, tokenizer):
    """
    Hàm tokenization cho mỗi batch.
    Input: một example dict chứa "text" và "label".
    Output: dict with tokenized fields.
    """
    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_pred):
    """
    Tính accuracy dựa trên logits và labels, dùng thư viện 'evaluate'.
    """
    metric = load("accuracy")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return metric.compute(predictions=preds, references=labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Đường dẫn đến file config YAML"
    )
    args = parser.parse_args()

    # 1. Đọc cấu hình từ YAML
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model_name    = config["model_name"]
    data_dir      = config["data_dir"]
    output_dir    = config["output_dir"]
    batch_size    = int(config["batch_size"])
    lr            = float(config["learning_rate"]) 
    epochs        = int(config["epochs"])
    eval_steps    = int(config["eval_steps"])
    logging_steps = int(config["logging_steps"])


    os.makedirs(output_dir, exist_ok=True)

    # 2. Load tokenizer và model base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSequenceClassification.from_pretrained(
                    model_name, num_labels=2
                )

    # 3. Load dữ liệu IMDb
    train_ds, test_ds = load_imdb_data(data_dir)

    # 4. Tokenize datasets
    train_ds = train_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer), batched=True
    )
    test_ds  = test_ds.map(
        lambda ex: tokenize_fn(ex, tokenizer), batched=True
    )

    # 5. Data collator cho dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 6. Khởi W&B run
    run = wandb.init(
        project="imdb-sentiment",
        config={
            "model_name": model_name,
            "batch_size": batch_size,
            "learning_rate": lr,
            "epochs": epochs
        }
    )

    # 7. Thiết lập TrainingArguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        eval_steps=eval_steps,
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        report_to=["wandb"],
        run_name=run.name
    )

    # 8. Khởi HF Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 9. Bắt đầu train và evaluate
    trainer.train()
    eval_result = trainer.evaluate()
    print("Eval result:", eval_result)

    # 10. Lưu model & tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    run.finish()
    print(f"Training complete! Model saved to '{output_dir}'")

if __name__ == "__main__":
    main()
