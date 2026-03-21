import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("predictor")


def format_metrics(m, name=None):
    prefix = f"{name}: " if name else ""
    return f"{prefix}F1={m['f1']:.4f}, Acc={m['accuracy']:.4f}, P={m['precision']:.4f}, R={m['recall']:.4f}"


class BertPredictor:
    def __init__(self, model_name, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.model_dir = self.models_dir / model_name
        
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model not found: {self.model_dir}")
        
        LOGGER.info(f"Loading model: {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        LOGGER.info(f"Model loaded on {self.device}")

    def predict(self, text):
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.config["model"]["max_length"],
            return_tensors="pt"
        )
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item()
        
        return {
            "label": pred,
            "confidence": round(confidence, 4),
        }

    def predict_texts(self, texts):
        results = []
        for text in texts:
            results.append(self.predict(str(text)))
        return results

    def predict_batch(self, csv_path, output_path=None):
        LOGGER.info(f"Predicting batch: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if "text" not in df.columns:
            raise ValueError("CSV must have 'text' column")
        
        results = self.predict_texts(df["text"].tolist())
        df["predict"] = [r["label"] for r in results]
        df["confidence"] = [r["confidence"] for r in results]
        
        if output_path:
            df.to_csv(output_path, index=False)
            LOGGER.info(f"Predictions saved: {output_path}")
        
        return df

    def evaluate(self, csv_path):
        LOGGER.info(f"Evaluating on: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have 'text' and 'label' columns")
        
        predictions = []
        for text in df["text"]:
            result = self.predict(str(text))
            predictions.append(result["label"])
        
        labels = df["label"].tolist()
        
        return {
            "dataset": Path(csv_path).name,
            "n_samples": len(df),
            "accuracy": round(accuracy_score(labels, predictions), 4),
            "f1": round(f1_score(labels, predictions), 4),
            "precision": round(precision_score(labels, predictions), 4),
            "recall": round(recall_score(labels, predictions), 4),
        }
