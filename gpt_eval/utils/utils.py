import json
import os
from typing import List
from datetime import datetime
from gpt_eval.config.config_models import DatasetConfig, EvalPrompt, EvalResponse


def ensure_directory_exists(dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path


def save_evaluation_results(
    model_name: str,
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
            "response": eval_prompts[idx].model_response.response,
            **eval_prompts[idx].model_response.prompt.model_dump(),
        }
        data.append(
            {
                "model": model,
                "evaluator": item.model_dump(mode="json"),
            }
        )

    with open(
        os.path.join(model_results_dir, f"{clean_ds_name}-results.json"), "w+"
    ) as fn:
        json.dump(data, fn, indent=4)


def get_dataset_config(
    ds_id: str, ds_config_list: List[DatasetConfig]
) -> DatasetConfig:
    filtered_ds_configs = filter(lambda ds: ds.id == ds_id, ds_config_list)
    ds_config = next(filtered_ds_configs, None)
    # sanity check
    if not ds_config:
        raise ValueError("Unable to determine dataset config")

    return ds_config


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


def dump_configs(dir_path: str, configs):
    with open(dir_path / "config.json", "w+") as fn:
        data = {}
        for key, config in configs.items():
            if isinstance(config, list):
                data[key] = [c.model_dump(mode="json") for c in config]
            else:
                data[key] = config.model_dump(mode="json")

        json.dump(data, fn, indent=4)
