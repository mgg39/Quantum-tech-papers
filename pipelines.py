"""
pipelines.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module provides a command-line interface for various tasks related to the Quantum Tech Papers Dashboard.
It includes options for extracting data from README, enriching the dataset incrementally, and training models.

Functions:
    load_config(config_path="config.yaml") -> dict:
        Loads the configuration from a YAML file.

    main() -> None:
        Parses command-line arguments and executes the corresponding tasks.
"""
import argparse
import yaml
from app.streamlit_app import run_streamlit_app
from src.training import main as train_model_main
from src.enrich_papers_incremental import enrich_incrementally
from src.extract_papers_to_csv import extract_data_from_readme


def load_config(config_path="config.yaml") -> dict:
    """
    Loads the configuration from a YAML file.

    Parameters:
    config_path (str): The path to the configuration file. Default is "config.yaml".

    Returns:
    dict: A dictionary containing the configuration settings.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Parses command-line arguments and executes the corresponding tasks.

    Returns:
    None
    """
    parser = argparse.ArgumentParser(
        description="Quantum Tech Papers Dashboard")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract data from README.")
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="Fetch metadata from arXiv / CrossRef and update enriched CSV.")
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the model training pipeline (NER, n-grams, SBERT embeddings).")

    args = parser.parse_args()
    config = load_config()

    if args.extract:
        print("Extracting data from README...")
        extract_data_from_readme(
            config['paths']['readme_input'],
            config['paths']['input_csv'])
        print("Extraction completed successfully.")

    if args.enrich:
        print("Starting incremental enrichment...")
        enrich_incrementally(
            config['paths']['input_csv'],
            config['paths']['enriched_csv'])
        print("Enrichment completed successfully.")

    if args.train:
        print("Starting the model training pipeline...")
        train_model_main(config)
        print("Training completed successfully.")

    if not any([args.extract, args.enrich, args.train]):
        print("No valid option provided. Use --extract, --train, or --enrich.")


if __name__ == "__main__":
    main()
