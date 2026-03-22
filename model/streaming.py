import logging
import os
import random
import threading
import time
from queue import Queue, Empty
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, f1_score
from model.predictor import BertPredictor

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TRUMP_MODEL = "baseline_frozen_encoder_trump"

class StreamEmulator:
    def __init__(self, model_name, batch_sz=32, trump_threshold=0.8):
        self.model_name = model_name
        self.trump_threshold = trump_threshold
        self.model = BertPredictor(model_name)
        self.trump_model = BertPredictor(TRUMP_MODEL) if model_name != TRUMP_MODEL else None
        self.using_trump = False
        self.batch_sz = batch_sz
        self.queue = Queue()
        self.processed = []
        self.lock = threading.Lock()
        self.timings = []
        self.report_every = 1.0
        self.batch_timeout = 5.0
        self._stop = threading.Event()
        self._done = threading.Event()
        self._n = 0

    def _check_trump_drift(self, batch):
        if not self.trump_model:
            return
        trump_count = sum(1 for x in batch if "trump" in x["text"].lower())
        trump_ratio = trump_count / len(batch)
        if trump_ratio >= self.trump_threshold and not self.using_trump:
            logger.info(f"[DRIFT] trump={trump_ratio:.0%}, switching to {TRUMP_MODEL}")
            self.using_trump = True
        elif trump_ratio < self.trump_threshold and self.using_trump:
            logger.info(f"[DRIFT] trump={trump_ratio:.0%}, switching back to {self.model_name}")
            self.using_trump = False

    def _flush(self, batch):
        if not batch:
            return
        self._check_trump_drift(batch)
        model = self.trump_model if self.using_trump else self.model
        t0 = time.time()
        preds = model.predict_texts([x["text"] for x in batch])
        with self.lock:
            self.timings.append(time.time() - t0)
            for item, p in zip(batch, preds):
                item["pred"] = p["label"]
                self.processed.append(item)

    def _stats(self):
        with self.lock:
            if not self.processed:
                return f"[0/{self._n}] queue={self.queue.qsize()} | waiting..."
            y_t = [x["label"] for x in self.processed]
            y_p = [x["pred"] for x in self.processed]
            n_done = len(self.processed)
            avg_t = sum(self.timings) / len(self.timings) if self.timings else 0
        mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        return f"[{n_done}/{self._n}] queue={self.queue.qsize()} | F1={f1_score(y_t,y_p):.3f} Acc={accuracy_score(y_t,y_p):.3f} | batch_t={avg_t:.2f}s mem={mem:.0f}MB"

    def _reporter(self):
        while not self._stop.wait(self.report_every):
            logger.info(self._stats())

    def _consumer(self):
        batch = []
        t_last = time.time()
        while True:
            try:
                batch.append(self.queue.get(timeout=0.1))
                if len(batch) >= self.batch_sz or (batch and time.time() - t_last > self.batch_timeout):
                    self._flush(batch)
                    batch, t_last = [], time.time()
            except Empty:
                if self._done.is_set() and self.queue.empty():
                    if batch:
                        self._flush(batch)
                    break
                if batch and time.time() - t_last > self.batch_timeout:
                    self._flush(batch)
                    batch, t_last = [], time.time()

    def run(self, path, rps=10, n_samples=None):
        data = pd.read_csv(path)
        if n_samples:
            data = data.sample(n=min(n_samples, len(data)))
        self._n = len(data)
        delay = 1.0 / rps
        logger.info(f"stream: n={self._n}, rps={rps}, batch={self.batch_sz}")
        t0 = time.time()
        threading.Thread(target=self._consumer, daemon=True).start()
        threading.Thread(target=self._reporter, daemon=True).start()
        for _, row in data.iterrows():
            self.queue.put({"text": row["text"], "label": row.get("label")})
            time.sleep(delay * random.uniform(0.5, 1.5))
        self._done.set()
        while not self.queue.empty() or len(self.processed) < self._n:
            time.sleep(0.1)
        self._stop.set()
        logger.info(self._stats())
        logger.info(f"done: {self._n} msgs in {time.time()-t0:.1f}s")
        return self.processed
