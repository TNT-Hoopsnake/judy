import json
import os
from typing import List
from types import ModuleType
import openai
from easyllm.clients import huggingface
from gpt_eval.config.config_models import DatasetConfig
from gpt_eval.config import ApiTypes


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


def save_evaluation_results(model_name: str, dataset_name: str, data: dict, results_dir: str):
    ensure_directory_exists(results_dir)

    model_results_dir = ensure_directory_exists(os.path.join(results_dir, model_name))
    clean_ds_name = dataset_name.split("/")[-1]

    with open(
        os.path.join(model_results_dir, f"{clean_ds_name}-results.json"), "w+"
    ) as fn:
        json.dump(data, fn, indent=4)


def get_dataset_config(ds_name: str, ds_config_list: List[DatasetConfig]) -> DatasetConfig:
    filtered_ds_configs = filter(lambda ds: ds.name == ds_name, ds_config_list)
    ds_config = next(filtered_ds_configs, None)
    # sanity check
    if not ds_config:
        raise ValueError("Unable to determine dataset config")

    return ds_config
