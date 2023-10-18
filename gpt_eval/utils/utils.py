from gpt_eval.config import ApiTypes
from easyllm.clients import huggingface
import openai
import os
from gpt_eval.config import RESULTS_DIR
import json

def get_completion_library(api_type, api_base):
    if api_type == ApiTypes.OPENAI:
        lib = openai
        # openai lib requires api_key to be set, even if we're not accessing the actual OAI api
        openai.api_key=""
    elif api_type == ApiTypes.TGI:
        lib= huggingface
    else:
        raise ValueError(f'Unable to determine completion library for api type: {api_type}')
    
    lib.api_base = api_base
    return lib

def ensure_directory_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path

def save_evaluation_results(model_name, dataset_name, data):
    ensure_directory_exists(RESULTS_DIR)
    
    model_results_dir = ensure_directory_exists(os.path.join(RESULTS_DIR, model_name))
    clean_ds_name = dataset_name.split('/')[-1]

    with open(os.path.join(model_results_dir, f'{clean_ds_name}-results.json'), 'w+') as fn:
        json.dump(data, fn, indent=4)


def get_dataset_config(ds_name, ds_config_list):
    filtered_ds_configs = filter(lambda ds: ds.name == ds_name, ds_config_list)
    ds_config = next(filtered_ds_configs, None)
    # sanity check
    if not ds_config:
        raise ValueError('Unable to determine dataset config')
    
    return ds_config