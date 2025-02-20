"""
config_loader.py

Author: Ricard Santiago Raigada García
Date: 15-02-2025

This module contains functions for loading configuration files and text resources for the application.

Functions:
    load_config(config_path='config.yaml') -> dict:
        Loads the configuration from a YAML file.

    load_texts(language='en', base_path='i18n') -> dict:
        Loads text resources from a JSON file based on the specified language.
"""
import yaml
import json


def load_config(config_path='config.yaml') -> dict:
    """
    Loads the configuration from a YAML file.

    Parameters:
    config_path (str): The path to the configuration file. Default is 'config.yaml'.

    Returns:
    dict: A dictionary containing the configuration settings.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_texts(language='en', base_path='i18n') -> dict:
    """
    Loads text resources from a JSON file based on the specified language.

    Parameters:
    language (str): The language code for the text resources. Default is 'en'.
    base_path (str): The base path to the directory containing the text resource files. Default is 'i18n'.

    Returns:
    dict: A dictionary containing the text resources.
    """
    path = f"{base_path}/{language}.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
