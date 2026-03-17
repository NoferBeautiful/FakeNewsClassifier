import json
import logging
import shutil
from pathlib import Path

import click
import kagglehub


LOGGER = logging.getLogger("download_data")

DATASETS = {
    "saurabhshahane/fake-news-classification": "fake_news_classification",
    "clmentbisaillon/fake-and-real-news-dataset": "fake_and_real_news_dataset",
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def download_one(dataset_ref, dataset_alias, output_root):
    cache_path = Path(kagglehub.dataset_download(dataset_ref))
    target_path = output_root / dataset_alias
    target_path.mkdir(parents=True, exist_ok=True)

    for item in cache_path.iterdir():
        destination = target_path / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)

    return {
        "dataset": dataset_ref,
        "cache_path": str(cache_path),
        "target_path": str(target_path),
    }


@click.command()
@click.option("--output-dir", default="./data/raw", show_default=True, type=click.Path(path_type=Path))
def main(output_dir):
    setup_logging()
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for dataset_ref, dataset_alias in DATASETS.items():
        LOGGER.info(f"Downloading {dataset_ref}")
        info = download_one(dataset_ref, dataset_alias, output_root)
        manifest.append(info)
        LOGGER.info(f"Saved to {info['target_path']}")

    manifest_path = output_root / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LOGGER.info(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
