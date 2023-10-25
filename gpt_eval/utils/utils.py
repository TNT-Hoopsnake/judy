import json
import os
from typing import List
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
        data.append(
            {
                "model": eval_prompts[idx].model_response.model_dump(),
                "evaluator": item.model_dump(),
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
