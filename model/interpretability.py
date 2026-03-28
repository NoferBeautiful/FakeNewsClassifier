from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer

STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "must", "shall", "can", "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "and", "but", "or", "nor", "so", "yet", "not", "only", "than", "too", "very", "just", "also", "that", "this", "these", "those", "it", "its", "they", "them", "their", "he", "she", "him", "her", "his", "we", "us", "our", "you", "your", "who", "whom", "which", "what", "where", "when", "why", "how", "all", "each", "every", "any", "some", "no", "more", "most", "other"}


class AttentionInterpreter:
    def __init__(self, model_name):
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        model_dir = Path(self.config["paths"]["models_dir"]) / model_name
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir), output_attentions=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def explain(self, text):
        data = self.explain_detailed(text)
        lines = [f"Prediction: {'FAKE' if data['label']==1 else 'REAL'} ({data['confidence']:.2%})", "",
                 f"Punctuation: {data['punct']:.4f}", f"Stop words: {data['stop']:.4f}", "", "Top words:"]
        for token, score in data['top_words'][:10]:
            lines.append(f"  {token}: {score:.4f}")
        return "\n".join(lines)

    def explain_detailed(self, text):
        inputs = self.tokenizer(text, truncation=True, max_length=self.config["model"]["max_length"], return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        attn = outputs.attentions[-1].mean(dim=1).squeeze(0)[0, 1:-1].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
        punct, stop, content = 0.0, 0.0, []
        token_scores = []
        for token, score in zip(tokens, attn.tolist()):
            clean = token[2:] if token.startswith("##") else token
            token_scores.append((token, score))
            if not clean.isalpha():
                punct += score
            elif clean.lower() in STOP_WORDS:
                stop += score
            else:
                content.append((token, score))
        content.sort(key=lambda x: x[1], reverse=True)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
        return {
            "label": pred,
            "confidence": probs[0][pred].item(),
            "punct": punct,
            "stop": stop,
            "top_words": content,
            "all_tokens": token_scores,
        }
