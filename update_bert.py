import logging

import click

from train_bert import DEFAULT_OUTPUT_DIR, setup_logging, update_pipeline


LOGGER = logging.getLogger("update_bert")


@click.command()
@click.option("--new-data-csv", required=True, type=click.Path(exists=True))
@click.option("--model-dir", required=True, type=click.Path(exists=True))
@click.option("--output-dir", default=DEFAULT_OUTPUT_DIR, show_default=True, type=click.Path())
def main(new_data_csv, model_dir, output_dir):
    setup_logging()
    final_path = update_pipeline(new_data_csv, model_dir, output_dir)
    LOGGER.info(f"Update completed. Artifacts saved to: {final_path}")


if __name__ == "__main__":
    main()
