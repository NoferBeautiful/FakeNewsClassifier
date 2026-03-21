import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset, concatenate_datasets
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
from model.predictor import format_metrics

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("trainer")


class BertTrainer:
    def __init__(self):
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        set_seed(self.config["training"]["seed"])

    def compute_metrics(self, eval_pred):
        preds = np.argmax(eval_pred[0], axis=-1)
        labels = eval_pred[1]
        return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds),
                "precision": precision_score(labels, preds), "recall": recall_score(labels, preds)}

    def _load_dataset(self, csv_path, tokenizer, weight=None):
        LOGGER.info(f"Loading: {csv_path}")
        df = pd.read_csv(csv_path)[["text", "label"]].dropna()
        df["text"] = df["text"].astype(str)
        df = df.rename(columns={"label": "labels"})
        if weight and weight != 1.0:
            df = df.sample(n=int(len(df) * weight), random_state=self.config["training"]["seed"])
        dataset = Dataset.from_pandas(df, preserve_index=False)
        return dataset.map(lambda b: tokenizer(b["text"], truncation=True, max_length=self.config["model"]["max_length"]), batched=True)

    def _resolve_datasets(self, datasets_config):
        if datasets_config is None:
            datasets_config = self.config["training"].get("datasets", {})
        if isinstance(datasets_config, str):
            preset_name = datasets_config
            datasets_config = self.config["dataset_presets"][preset_name]
            LOGGER.info(f"Using preset: {preset_name}")
        return datasets_config

    def train(self, name, base_model=None, datasets_config=None):
        datasets_config = self._resolve_datasets(datasets_config)
        cfg = self.config["training"]
        
        if base_model:
            LOGGER.info(f"Updating: {base_model} -> {name}")
            base_dir = self.models_dir / base_model
            tokenizer = AutoTokenizer.from_pretrained(str(base_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(base_dir))
        else:
            LOGGER.info(f"Training: {name}")
            tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["name"])
            model = AutoModelForSequenceClassification.from_pretrained(self.config["model"]["name"], num_labels=2)
        
        if cfg.get("freeze_encoder"):
            for p in model.distilbert.parameters():
                p.requires_grad = False
            LOGGER.info("Encoder frozen")
        
        train_ds = [self._load_dataset(i["path"] if isinstance(i, dict) else i, tokenizer, 
                    i.get("weight", 1.0) if isinstance(i, dict) else 1.0) for i in datasets_config["train"]]
        train_dataset = concatenate_datasets(train_ds).shuffle(seed=cfg["seed"])
        LOGGER.info(f"Train: {len(train_dataset)} samples")
        
        eval_datasets = {Path(i).stem: self._load_dataset(i, tokenizer) for i in datasets_config.get("eval", [])}
        first_eval = list(eval_datasets.values())[0] if eval_datasets else train_dataset.train_test_split(test_size=0.1)["test"]
        
        trainer = Trainer(
            model=model, processing_class=tokenizer,
            args=TrainingArguments(
                output_dir=str(self.models_dir / name / "checkpoints"),
                learning_rate=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0)),
                num_train_epochs=cfg["num_epochs"], per_device_train_batch_size=cfg["batch_size"],
                per_device_eval_batch_size=cfg["batch_size"], eval_strategy="epoch", save_strategy="epoch",
                load_best_model_at_end=True, metric_for_best_model="f1", save_total_limit=2, report_to=[], seed=cfg["seed"],
                label_smoothing_factor=float(cfg.get("soft_labeling", 0.0)),
            ),
            train_dataset=train_dataset, eval_dataset=first_eval,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        trainer.train()
        
        eval_results = {}
        for eval_name, eval_ds in eval_datasets.items():
            metrics = trainer.evaluate(eval_ds)
            eval_results[eval_name] = {k: round(v, 4) for k, v in metrics.items()}
            LOGGER.info(f"  {format_metrics({k.replace('eval_', ''): v for k, v in metrics.items() if k.startswith('eval_')}, eval_name)}")
        
        model_dir = self.models_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        with open(model_dir / "metrics.json", "w") as f:
            json.dump({"name": name, "metrics": eval_results.get(list(eval_results.keys())[0], {}), "eval_results": eval_results}, f, indent=2)
        LOGGER.info(f"Saved: {model_dir}")
        return model_dir

    def update(self, name, base_model, datasets_config=None):
        return self.train(name, base_model=base_model, datasets_config=datasets_config)
