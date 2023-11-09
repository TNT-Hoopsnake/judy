import json
import os
import pathlib
import sys
from typing import List
from datetime import datetime
from judy.config import (
    DatasetConfig,
    TaskConfig,
    EvaluatedModel,
    get_config_definitions,
)
from judy.config.settings import DATASET_CONFIG_PATH, EVAL_CONFIG_PATH, RUN_CONFIG_PATH
from judy.config.validator import load_and_validate_configs
from judy.responders import EvalPrompt, EvalResponse
from judy.config.logging import logger as log


def matches_tag(config: TaskConfig | EvaluatedModel | DatasetConfig, tag: str) -> bool:
    if not tag or not hasattr(config, "tags"):
        return True
    config_tags = config.tags or []
    if tag in config_tags:
        return True
    return False


def ensure_directory_exists(dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        log.info("Created new directory: %s", dir_path)

    return dir_path


def save_evaluation_results(
    model_name: str,
    scenario_id: str,
    task_id: str,
    dataset_name: str,
    eval_prompts: List[EvalPrompt],
    eval_results: List[EvalResponse],
    results_dir: str,
):
    ensure_directory_exists(results_dir)

    model_results_dir = ensure_directory_exists(os.path.join(results_dir, model_name))
    clean_ds_name = dataset_name.split("/")[-1]
    data = []
    for idx, item in enumerate(eval_results):
        model = {
            "response": eval_prompts[idx].response_data.response,
            **eval_prompts[idx].response_data.prompt.model_dump(),
        }
        data.append(
            {
                "task_id": task_id,
                "scenario_id": scenario_id,
                "model": model,
                "evaluator": item.model_dump(mode="json"),
            }
        )

    with open(
        os.path.join(model_results_dir, f"{clean_ds_name}-results.json"), "w+"
    ) as fn:
        json.dump(data, fn, indent=4)


def dump_metadata(
    dir_path: str, dataset_tags: List[str], task_tags: List[str], model_tags: List[str]
):
    with open(dir_path / "metadata.json", "w+") as fn:
        data = {
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
    # ensure paths to config files are valid
    try:
        for config_path in [eval_config_path, dataset_config_path, run_config_path]:
            if config_path and not pathlib.Path(config_path).is_file():
                raise FileNotFoundError(
                    f"Config file is not a valid file or does not exist at path: {config_path}"
                )
    except FileNotFoundError as e:
        log.error(str(e))
        sys.exit(1)

    eval_config_path = eval_config_path or EVAL_CONFIG_PATH
    dataset_config_path = dataset_config_path or DATASET_CONFIG_PATH
    run_config_path = run_config_path or RUN_CONFIG_PATH

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
    ensure_directory_exists(results_dir)
    log.info("Results directory: %s", results_dir)

    return results_dir
