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
        self.seed = self.config.get("seed", 42)
        self.rps = self.config.get("rps", 1)
        
        self.raw_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)

    def download(self) -> Path:
        LOGGER.info(f"Downloading: {self.config['dataset']}")
        try:
            path = Path(kagglehub.dataset_download(self.config["dataset"]))
            LOGGER.info(f"Downloaded to: {path}")
            return path
        except Exception as e:
            LOGGER.error(f"Download failed: {e}")
            raise

    def load_raw_data(self, data_path: Path) -> pd.DataFrame:
        LOGGER.info("Loading raw data")
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
            
            train_df.to_csv(self.processed_path / f"train{name}.csv", index=False)
            test_df.to_csv(self.processed_path / f"test{name}.csv", index=False)
            
            LOGGER.info(f"Split {name}: train{name}={len(train_df)}, test{name}={len(test_df)}")

    def collect(self):
        LOGGER.info("Starting data collection")
        LOGGER.info(f"Config: seed={self.seed}, rps={self.rps}")
        data_path = self.download()
        df = self.load_raw_data(data_path)
        df = self.preprocess(df)
        batches = self.split_into_batches(df)
        self.save_batches(batches)
        self.save_splits(df)
        LOGGER.info("Done")
        return df


if __name__ == "__main__":
    collector = DataCollector()
    collector.collect()
