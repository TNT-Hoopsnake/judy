import os
from datasets import load_from_disk, load_dataset
import numpy as np
from .formatter import DATASET_MAPPER
from gpt_eval.utils.config import RANDOM_SEED

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

# replace with env var
np.random.seed(RANDOM_SEED)

def load_formatted_data(num_idxs, ds_name, ds_vers=None):
    dataset = get_dataset(ds_name, ds_vers)

    format_func = DATASET_MAPPER.get(ds_name)
    if format_func:
        eval_idxs = get_eval_idxs(num_idxs, len(dataset['train']))
        return format_func(dataset, eval_idxs)
    else:
        raise ValueError(f"Unable to map dataset ({ds_name}) to formatting function")
    

def download_dataset(dataset, version=None, force=False):
    if not os.path.exists(DATASETS_DIR):
        os.mkdir(DATASETS_DIR)

    dataset_path = os.path.join(DATASETS_DIR, dataset.split('/')[-1])
    
    # dont redownload the dataset unless the user forces it
    if force or not os.path.exists(dataset_path):
        print(f"Downloading dataset: ({dataset})")
        dataset = load_dataset(
            dataset,
            version,
        )
        dataset.save_to_disk(dataset_path)
    else:
        print(f"Dataset ({dataset}) already exists. Set force=True to redownload")


def get_dataset(ds_name, ds_vers):
    dataset_path = os.path.join(DATASETS_DIR, ds_name.split('/')[-1])
    if not os.path.exists(dataset_path):
        download_dataset(ds_name, ds_vers)
    
    # Load the specified dataset
    dataset = load_from_disk(dataset_path)

    return dataset


def get_eval_idxs(num_idxs, max_idx):
    return np.random.randint(low=0, high=max_idx, size=num_idxs)




if __name__ == "__main__":
    data = [
        ('ms_marco','v1.1'),
        ('cnn_dailymail', '3.0.0'),
        ('dim/mt_bench_en', None),
        ('xsum', None),
        ('quac', None),
        ('squad_v2', None)
    ]
    for ds, version in data:
        download_dataset(ds, version)


