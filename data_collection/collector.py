import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
import kagglehub
from dateutil import parser as date_parser

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
LOGGER = logging.getLogger("data_collection")


class DataCollector:
    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        with open(config_path) as f:
            self.config = yaml.safe_load(f)["data_collection"]
        
        self.raw_path = Path(self.config["raw_path"])
        self.processed_path = Path(self.config["processed_path"])
        self.meta_path = Path(self.config["meta_path"])
        self.seed = self.config.get("seed", 42)
        self.rps = self.config.get("rps", 1)
        
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        self.meta_path.mkdir(parents=True, exist_ok=True)

    def get_data_from_source(self, source: dict) -> pd.DataFrame:
        source_type = source["type"]
        
        if source_type == "kaggle":
            dataset = source["dataset"]
            LOGGER.info(f"Downloading from Kaggle: {dataset}")
            path = Path(kagglehub.dataset_download(dataset))
            LOGGER.info(f"Downloaded to: {path}")
            return self.load_raw_data(path)
        
        elif source_type == "local":
            path = Path(source["path"])
            if not path.exists():
                raise FileNotFoundError(f"Local path not found: {path}")
            LOGGER.info(f"Loading from local: {path}")
            return self.load_raw_data(path)
        
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def load_raw_data(self, data_path: Path) -> pd.DataFrame:
        LOGGER.info(f"Loading raw data from {data_path}")
        try:
            df_fake = pd.read_csv(data_path / "Fake.csv")
            df_fake["label"] = 1
            df_true = pd.read_csv(data_path / "True.csv")
            df_true["label"] = 0
            df = pd.concat([df_fake, df_true], ignore_index=True)
            LOGGER.info(f"Loaded {len(df)} rows")
            return df
        except Exception as e:
            LOGGER.error(f"Load failed: {e}")
            raise

    def load_all_sources(self) -> pd.DataFrame:
        sources = self.config.get("sources", [])
        if not sources:
            raise ValueError("No sources configured")
        
        dfs = []
        for source in sources:
            df = self.get_data_from_source(source)
            dfs.append(df)
        
        combined = pd.concat(dfs, ignore_index=True)
        LOGGER.info(f"Combined {len(sources)} sources: {len(combined)} rows total")
        return combined

    def parse_date(self, d):
        try:
            return date_parser.parse(str(d))
        except:
            return None

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info("Preprocessing")
        df = df.copy()
        df["date"] = df["date"].apply(self.parse_date)
        df = df.dropna(subset=["text", "date"])
        df["text"] = df["text"].astype(str).str.strip()
        df["title"] = df["title"].fillna("").astype(str).str.strip()
        df = df.drop_duplicates(subset=["text"], keep="first")
        df = df.sort_values("date").reset_index(drop=True)
        LOGGER.info(f"After preprocessing: {len(df)} rows")
        return df

    def split_into_batches(self, df: pd.DataFrame) -> dict:
        LOGGER.info(f"Splitting into batches by {self.config['batch_freq']}")
        df = df.copy()
        df["batch"] = df["date"].dt.to_period(self.config["batch_freq"])
        batches = {}
        for batch_id, batch_df in df.groupby("batch"):
            batches[str(batch_id)] = batch_df.drop(columns=["batch"])
        LOGGER.info(f"Created {len(batches)} batches")
        return batches

    def save_batches(self, batches: dict):
        LOGGER.info("Saving batches")
        for batch_id, batch_df in batches.items():
            batch_df_save = batch_df.copy()
            batch_df_save["date"] = batch_df_save["date"].dt.strftime("%Y-%m-%d")
            batch_file = self.raw_path / f"batch_{batch_id}.csv"
            batch_df_save.to_csv(batch_file, index=False)
            LOGGER.info(f"Batch {batch_id}: {len(batch_df)} rows")

    def calculate_meta(self, name: str, df: pd.DataFrame) -> dict:
        text_lengths = df["text"].str.len()
        return {
            "name": name,
            "n_rows": len(df),
            "n_fake": int((df["label"] == 1).sum()),
            "n_real": int((df["label"] == 0).sum()),
            "fake_ratio": round((df["label"] == 1).mean(), 4),
            "text_len_mean": round(text_lengths.mean(), 2),
            "text_len_std": round(text_lengths.std(), 2),
            "text_len_min": int(text_lengths.min()),
            "text_len_max": int(text_lengths.max()),
            "created_at": datetime.now().isoformat(),
        }

    def save_meta(self, name: str, df: pd.DataFrame):
        meta = self.calculate_meta(name, df)
        meta_file = self.meta_path / f"{name}_meta.json"
        with open(meta_file, "w") as f:
            json.dump(meta, f, indent=2)
        LOGGER.info(f"Meta saved: {meta_file}")

    def add_arrival_time(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy().reset_index(drop=True)
        start_time = datetime.now()
        interval = timedelta(seconds=1.0 / self.rps)
        df["arrival_time"] = [
            (start_time + interval * i).isoformat() for i in range(len(df))
        ]
        return df

    def save_splits(self, df: pd.DataFrame):
        LOGGER.info("Creating train/test splits")
        from sklearn.model_selection import train_test_split
        
        splits_config = self.config.get("splits", [])
        if not splits_config:
            LOGGER.warning("No splits configured")
            return
        
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        
        remaining_df = df
        for split in splits_config:
            name = split["name"]
            end_date = split.get("end_date")
            test_size = split.get("test_size", 0.2)
            
            if end_date:
                end_dt = pd.to_datetime(end_date)
                split_df = remaining_df[remaining_df["date"] < end_dt]
                remaining_df = remaining_df[remaining_df["date"] >= end_dt]
            else:
                split_df = remaining_df
                remaining_df = pd.DataFrame()
            
            if len(split_df) == 0:
                LOGGER.warning(f"Split {name}: no data")
                continue
            
            split_df_save = split_df.copy()
            split_df_save["date"] = split_df_save["date"].dt.strftime("%Y-%m-%d")
            
            train_df, test_df = train_test_split(
                split_df_save, test_size=test_size, random_state=self.seed, stratify=split_df_save["label"]
            )
            
            test_df = self.add_arrival_time(test_df)
            
            train_name = f"train{name}"
            test_name = f"test{name}"
            
            train_df.to_csv(self.processed_path / f"{train_name}.csv", index=False)
            test_df.to_csv(self.processed_path / f"{test_name}.csv", index=False)
            
            self.save_meta(train_name, train_df)
            self.save_meta(test_name, test_df)
            
            LOGGER.info(f"Split {name}: {train_name}={len(train_df)}, {test_name}={len(test_df)}")

    def collect(self):
        LOGGER.info("Starting data collection")
        LOGGER.info(f"Config: seed={self.seed}, rps={self.rps}")
        df = self.load_all_sources()
        df = self.preprocess(df)
        batches = self.split_into_batches(df)
        self.save_batches(batches)
        self.save_splits(df)
        LOGGER.info("Done")
        return df


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect()
