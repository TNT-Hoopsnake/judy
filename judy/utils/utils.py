import json
import pathlib
import sys
import shutil
from typing import List
from datetime import datetime
from judy.config import (
    DatasetConfig,
    TaskConfig,
    EvaluatedModel,
    get_config_definitions,
)
from judy.config.validator import load_and_validate_configs
from judy.config.logging import logger as log


def matches_tag(config: TaskConfig | EvaluatedModel | DatasetConfig, tag: str) -> bool:
    if not tag or not hasattr(config, "tags"):
        return True
    config_tags = config.tags or []
    if tag in config_tags:
        return True
    return False


def ensure_directory_exists(dir_path: str | pathlib.Path, clear_if_exists=False) -> str:
    dir_path = pathlib.Path(dir_path)
    if not dir_path.exists():
        dir_path.mkdir()
        log.info("Created new directory: %s", dir_path)
    elif clear_if_exists:
        log.info("Clearing existing directory: %s", dir_path)
        shutil.rmtree(dir_path)
        dir_path.mkdir()

    return dir_path


def dump_metadata(
    dir_path: str,
    dataset_tags: List[str],
    task_tags: List[str],
    model_tags: List[str],
    model_id: str | None = None,
):
    with open(dir_path / "metadata.json", "w+") as fn:
        data = {
            "model_id": model_id,
            "dataset_tags": dataset_tags,
            "task_tags": task_tags,
            "model_tags": model_tags,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        json.dump(data, fn, indent=4)


def load_configs(
    eval_config_path: str | pathlib.Path,
    dataset_config_path: str | pathlib.Path,
    run_config_path: str | pathlib.Path,
) -> dict:
    """
    Load and validate configuration files for evaluation.

    Args:
        eval_config_path (str | pathlib.Path): Path to the evaluation config file.
        dataset_config_path (str | pathlib.Path): Path to the dataset config file.
        run_config_path (str | pathlib.Path): Path to the run config file.

    Returns:
        dict: A dictionary containing the loaded and validated configuration settings.

    Ensures that the paths to the specified config files are valid.
    If any of the paths is invalid or the file does not exist, a FileNotFoundError is raised.
    The function then logs the paths of the configuration files and proceeds to validate
    the configurations using the defined configuration definitions. The validated
    configurations are returned as a dictionary.
    """
    try:
        for config_path in [eval_config_path, dataset_config_path, run_config_path]:
            if config_path and not pathlib.Path(config_path).is_file():
                raise FileNotFoundError(
                    f"Config file is not a valid file or does not exist at path: {config_path}"
                )
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    log.info("Evaluation config path: %s", eval_config_path)
    log.info("Dataset config path: %s", dataset_config_path)
    log.info("Run config path: %s", run_config_path)

    # Validate configs
    config_definitions = get_config_definitions(
        eval_config_path, dataset_config_path, run_config_path
    )
    return load_and_validate_configs(config_definitions)


def get_output_directory(output: str | pathlib.Path, run_name: str) -> pathlib.Path:
    # Check output directory
    try:
        output_dir = pathlib.Path(output)
        if not output_dir.is_dir():
            raise FileNotFoundError(f"Output directory does not exist: {output}")
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    results_dir = output_dir / run_name
    log.info("Results directory: %s", results_dir)

    return ensure_directory_exists(results_dir)
