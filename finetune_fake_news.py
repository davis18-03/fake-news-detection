import logging
import os
from typing import Tuple

import torch
import yaml
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

from src.data_utils import load_data, shuffle_data
from src.model_utils import get_device_info


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def split_data(df, test_size=0.2, seed=42) -> Tuple:
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)
    config = load_config()
    data_dir = config["data_dir"]
    model_name = config.get("model_name", "roberta-base")
    output_dir = config.get("finetuned_model_dir", "./finetuned_model")
    num_labels = 2
    epochs = config.get("epochs", 2)
    batch_size = config.get("batch_size", 2)
    max_length = config.get("max_length", 256)
    seed = config.get("seed", 42)

    logger.info("Loading and preparing data...")
    df = load_data(data_dir)
    df = shuffle_data(df, seed=seed)
    train_df, test_df = split_data(df, seed=seed)
    train_ds = Dataset.from_pandas(train_df)
    test_ds = Dataset.from_pandas(test_df)
    dataset = DatasetDict({"train": train_ds, "test": test_ds})

    logger.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    logger.info("Tokenizing data...")
    tokenized_datasets = dataset.map(
        lambda x: tokenizer(
            x["text"], padding="max_length", truncation=True, max_length=max_length
        ),
        batched=True,
    )
    tokenized_datasets = (
        tokenized_datasets.remove_columns(["text", "__index_level_0__"])
        if "__index_level_0__" in tokenized_datasets["train"].features
        else tokenized_datasets.remove_columns(["text"])
    )
    tokenized_datasets.set_format("torch")

    logger.info("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
        seed=seed,
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import (accuracy_score,
                                     precision_recall_fscore_support)

        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Evaluating on test set...")
    results = trainer.evaluate()
    logger.info(f"Test results: {results}")

    logger.info(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuning complete!")
    logger.info(f"Device: {get_device_info()}")


if __name__ == "__main__":
    main()
