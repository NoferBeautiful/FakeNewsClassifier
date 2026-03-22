import json
from pathlib import Path


class ModelRegistry:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)

    def list(self):
        if not self.models_dir.exists():
            return []
        return sorted([p.name for p in self.models_dir.iterdir() if (p / "metrics.json").exists()])

    def _load_json(self, name):
        with open(self.models_dir / name / "metrics.json") as f:
            return json.load(f)

    def _get_metrics(self, name):
        return self._load_json(name).get("metrics", {})

    def summary(self):
        models = []
        all_datasets = set()
        
        for name in self.list():
            data = self._load_json(name)
            eval_results = data.get("eval_results", {})
            all_datasets.update(eval_results.keys())
            
            model_info = {"name": name, "datasets": {}}
            for ds_name, ds_metrics in eval_results.items():
                model_info["datasets"][ds_name] = {
                    "f1": round(ds_metrics.get("eval_f1", 0), 4),
                    "accuracy": round(ds_metrics.get("eval_accuracy", 0), 4)
                }
            models.append(model_info)
        
        return {"total_models": len(models), "models": models, "datasets": sorted(all_datasets)}
