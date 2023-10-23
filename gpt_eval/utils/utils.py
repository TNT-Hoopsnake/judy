import json
from datetime import datetime
import os
from typing import List
from types import ModuleType
import openai
from easyllm.clients import huggingface
from gpt_eval.config.config_models import DatasetConfig
from gpt_eval.config import ApiTypes


def dump_metadata(dir_path, dataset_tags, scenario_tags, model_tags):
    with open(dir_path / "metadata.json", "w+") as fn:
        data = {
            "dataset_tags":dataset_tags,
            "scenario_tags":scenario_tags,
            "model_tags":model_tags,
            "timestamp":datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        json.dump(data, fn, indent=4)

def dump_configs(dir_path, configs):
    with open(dir_path / "config.json", "w+") as fn:
        data = {}
        for key, config in configs.items():
            if isinstance(config, list):
                data[key] = [c.model_dump(mode="json") for c in config]
            else:
                data[key] = config.model_dump(mode="json")

        json.dump(data, fn, indent=4)

def get_completion_library(api_type: ApiTypes, api_base: str) -> ModuleType:
    if api_type == ApiTypes.OPENAI:
        lib = openai
        # openai lib requires api_key to be set, even if we're not accessing the actual OAI api
        openai.api_key = ""
    elif api_type == ApiTypes.TGI:
        lib = huggingface
    else:
        raise ValueError(
            f"Unable to determine completion library for api type: {api_type}"
        )
    lib.api_base = api_base
    return lib


def ensure_directory_exists(dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path


def save_evaluation_results(
    model_name: str, dataset_name: str, data: dict, results_dir: str
):
    model_results_dir = ensure_directory_exists(os.path.join(results_dir, model_name))
    clean_ds_name = dataset_name.split("/")[-1]

    with open(
        os.path.join(model_results_dir, f"{clean_ds_name}-results.json"), "w+"
    ) as fn:
        json.dump(data, fn, indent=4)


def get_dataset_config(
    ds_name: str, ds_config_list: List[DatasetConfig]
) -> DatasetConfig:
    filtered_ds_configs = filter(lambda ds: ds.name == ds_name, ds_config_list)
    ds_config = next(filtered_ds_configs, None)
    # sanity check
    if not ds_config:
        raise ValueError("Unable to determine dataset config")

    return ds_config
