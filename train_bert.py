import json
import logging
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


LOGGER = logging.getLogger("train_bert")

SEED = 1337
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
DEFAULT_OUTPUT_DIR = "./models/bert"

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
EVAL_SIZE = 0.1
MAX_LENGTH = None
UNFREEZE_LAST_N_LAYERS = 2

TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 2
EARLY_STOPPING_PATIENCE = 2
LOGGING_STEPS = 50

TRAIN_PHASES = [
    {"name": "phase1", "epochs": 2.0, "learning_rate": 3e-4, "mode": "head_only"},
    {"name": "phase2", "epochs": 2.0, "learning_rate": 2e-5, "mode": "last_layers"},
]

UPDATE_PHASE = {"name": "update", "epochs": 1.0, "learning_rate": 1e-5, "mode": "last_layers"}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
    }


def load_and_prepare_dataset(csv_path, tokenizer):
    df = pd.read_csv(csv_path)
    required = {TEXT_COLUMN, LABEL_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="raise").astype(int)
    df = df.rename(columns={TEXT_COLUMN: "text", LABEL_COLUMN: "labels"})

    dataset = Dataset.from_pandas(df, preserve_index=False)
    splits = dataset.shuffle(seed=SEED).train_test_split(test_size=EVAL_SIZE, seed=SEED)

    def tokenize_batch(batch):
        kwargs = {"truncation": True}
        if MAX_LENGTH is not None:
            kwargs["max_length"] = MAX_LENGTH
        return tokenizer(batch["text"], **kwargs)

    return splits.map(tokenize_batch, batched=True)


def unfreeze_last_distilbert_layers(model, n_layers):
    if n_layers <= 0:
        return

    layer_ids = set()
    for name, _ in model.named_parameters():
        parts = name.split(".")
        if len(parts) > 3 and parts[0] == "distilbert" and parts[1] == "transformer" and parts[2] == "layer":
            try:
                layer_ids.add(int(parts[3]))
            except ValueError:
                continue

    if not layer_ids:
        return

    selected_ids = sorted(layer_ids)[-n_layers:]
    for name, param in model.named_parameters():
        for layer_id in selected_ids:
            if f"transformer.layer.{layer_id}." in name:
                param.requires_grad = True
                break


def set_trainable_params(model, mode):
    for _, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if "classifier" in name or "pre_classifier" in name:
            param.requires_grad = True

    if mode == "last_layers":
        unfreeze_last_distilbert_layers(model, UNFREEZE_LAST_N_LAYERS)


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def build_training_args(run_dir, epochs, learning_rate):
    return TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=SAVE_TOTAL_LIMIT,
        logging_strategy="steps",
        logging_steps=LOGGING_STEPS,
        report_to=[],
        seed=SEED,
    )


def run_phase(model, tokenizer, splits, run_dir, phase):
    set_trainable_params(model, phase["mode"])
    trainable, total = count_trainable_params(model)
    LOGGER.info(f"{phase['name']} trainable params: {trainable} / {total}")

    trainer = Trainer(
        model=model,
        args=build_training_args(run_dir / phase["name"], phase["epochs"], phase["learning_rate"]),
        train_dataset=splits["train"],
        eval_dataset=splits["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()
    metrics = trainer.evaluate()
    return trainer, metrics


def make_run_dir(output_dir):
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / f"v_{timestamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = base_dir / f"v_{timestamp}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_artifacts(model, tokenizer, run_dir):
    final_dir = run_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    torch.save(model.state_dict(), final_dir / "model.pth")
    return final_dir


def save_metrics(run_dir, params, metrics, final_dir):
    payload = {
        "params": params,
        "metrics": metrics,
        "final_model_dir": str(final_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def train_pipeline(train_csv, output_dir):
    set_seed(SEED)

    run_dir = make_run_dir(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    splits = load_and_prepare_dataset(train_csv, tokenizer)

    metrics = {}
    trainer = None
    for phase in TRAIN_PHASES:
        trainer, phase_metrics = run_phase(model, tokenizer, splits, run_dir, phase)
        metrics[phase["name"]] = phase_metrics

    final_dir = save_artifacts(model, tokenizer, run_dir)
    save_metrics(
        run_dir=run_dir,
        params={"mode": "train", "seed": SEED, "train_csv": train_csv, "model_name": MODEL_NAME},
        metrics=metrics,
        final_dir=final_dir,
    )

    return final_dir


def update_pipeline(new_data_csv, model_dir, output_dir):
    set_seed(SEED)

    run_dir = make_run_dir(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    splits = load_and_prepare_dataset(new_data_csv, tokenizer)

    _, update_metrics = run_phase(model, tokenizer, splits, run_dir, UPDATE_PHASE)

    final_dir = save_artifacts(model, tokenizer, run_dir)
    save_metrics(
        run_dir=run_dir,
        params={"mode": "update", "seed": SEED, "new_data_csv": new_data_csv, "source_model_dir": model_dir},
        metrics={"update": update_metrics},
        final_dir=final_dir,
    )

    return final_dir


@click.command()
@click.option("--train-csv", required=True, type=click.Path(exists=True))
@click.option("--output-dir", default=DEFAULT_OUTPUT_DIR, show_default=True, type=click.Path())
def main(train_csv, output_dir):
    setup_logging()
    final_path = train_pipeline(train_csv, output_dir)
    LOGGER.info(f"Training completed. Artifacts saved to: {final_path}")


if __name__ == "__main__":
    main()

