import os
from box.exceptions import BoxValueError
import yaml
from image_classifier import logger
import json
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path

@ensure_annotations
def read_yaml(yaml_path: Path) -> ConfigBox:
    try:
        with open(yaml_path) as yaml_file:
            content =yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {yaml_path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_dir(dir_path: list, verbose=True):
    for path in dir_path:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created:{path}")

@ensure_annotations
def save_json(path: Path, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"json file saved at: {path}")
