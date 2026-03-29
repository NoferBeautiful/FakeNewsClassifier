import logging

import click

from model.trainer import BertTrainer
from model.predictor import BertPredictor, format_metrics
from model.interpretability import AttentionInterpreter
from model.registry import ModelRegistry
from model.streaming import StreamEmulator

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger("run")


@click.group()
def cli():
    pass


@cli.command()
@click.option("--name", required=True)
def train(name: str):
    trainer = BertTrainer()
    model_path = trainer.train(name)
    LOGGER.info(f"Model saved: {model_path}")


@cli.command()
@click.option("--name", required=True)
@click.option("--base-model", required=True)
def update(name: str, base_model: str):
    trainer = BertTrainer()
    model_path = trainer.update(name, base_model)
    LOGGER.info(f"Model saved: {model_path}")


@cli.command()
@click.option("--model", required=True)
@click.option("--file", required=True, type=click.Path(exists=True))
@click.option("--output", default=None)
def inference(model: str, file: str, output: str):
    predictor = BertPredictor(model)
    
    if output is None:
        output = file.replace(".csv", "_predictions.csv")
    
    df = predictor.predict_batch(file, output)
    LOGGER.info(f"Predictions saved: {output}")
    LOGGER.info(f"Total: {len(df)}, Fake: {(df['predict'] == 1).sum()}, Real: {(df['predict'] == 0).sum()}")


@cli.command()
@click.option("--model", required=True)
@click.option("--files", required=True, multiple=True, type=click.Path(exists=True))
def validate(model: str, files: tuple):
    predictor = BertPredictor(model)
    
    LOGGER.info(f"Validating model '{model}' on {len(files)} datasets")
    
    results = []
    for file in files:
        result = predictor.evaluate(file)
        results.append(result)
        LOGGER.info(f"  {format_metrics(result, result['dataset'])}")
    
    if len(results) > 1:
        avg = {k: sum(r[k] for r in results) / len(results) for k in ["f1", "accuracy", "precision", "recall"]}
        LOGGER.info(f"  {format_metrics(avg, 'Average')}")


@cli.command()
@click.option("--model", required=True)
@click.option("--text", required=True)
def predict(model: str, text: str):
    predictor = BertPredictor(model)
    result = predictor.predict(text)
    label = "FAKE" if result["label"] == 1 else "REAL"
    LOGGER.info(f"Prediction: {label}")
    LOGGER.info(f"Confidence: {result['confidence']:.2%}")


@cli.command()
@click.option("--model", required=True)
@click.option("--text", required=True)
def explain(model: str, text: str):
    interpreter = AttentionInterpreter(model)
    explanation = interpreter.explain(text)
    LOGGER.info(f"\n{explanation}")


@cli.command()
def summary():
    registry = ModelRegistry()
    result = registry.summary()
    
    LOGGER.info(f"Total models: {result['total_models']}")
    
    for ds in result["datasets"]:
        LOGGER.info(f"\n{ds}:")
        for m in result["models"]:
            if ds in m["datasets"]:
                metrics = m["datasets"][ds]
                LOGGER.info(f"  {m['name']}: F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")


@cli.command()
@click.option("--model1", required=True)
@click.option("--model2", required=True)
@click.option("--dataset", required=True, type=click.Path(exists=True))
def compare(model1: str, model2: str, dataset: str):
    LOGGER.info(f"Comparing models on {dataset}")
    
    p1 = BertPredictor(model1)
    m1 = p1.evaluate(dataset)
    LOGGER.info(f"  {format_metrics(m1, model1)}")
    
    p2 = BertPredictor(model2)
    m2 = p2.evaluate(dataset)
    LOGGER.info(f"  {format_metrics(m2, model2)}")


@cli.command()
@click.option("--model", required=True)
@click.option("--file", required=True, type=click.Path(exists=True))
@click.option("--rps", default=10, type=int)
@click.option("--batch-size", default=32, type=int)
@click.option("--limit", default=None, type=int)
@click.option("--report-interval", default=1.0, type=float)
@click.option("--batch-timeout", default=5.0, type=float)
@click.option("--trump-threshold", default=0.5, type=float, help="Trump ratio to trigger model switch")
@click.option("--empty-ratio", default=0.0, type=float, help="Ratio of empty requests (0-1)")
def stream(model: str, file: str, rps: int, batch_size: int, limit: int, report_interval: float, batch_timeout: float, trump_threshold: float, empty_ratio: float):
    emu = StreamEmulator(model, batch_sz=batch_size, trump_threshold=trump_threshold)
    emu.report_every = report_interval
    emu.batch_timeout = batch_timeout
    emu.run(file, rps=rps, n_samples=limit, empty_ratio=empty_ratio)


if __name__ == "__main__":
    cli()
