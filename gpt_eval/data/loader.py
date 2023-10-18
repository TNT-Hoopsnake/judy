import os
from datasets import load_from_disk, load_dataset
import numpy as np
from gpt_eval.config import DatasetConfig, DATASETS_DIR, SourceTypes
import gpt_eval.data.formatters as formatters
from gpt_eval.utils import ensure_directory_exists

def load_formatted_data(ds_config: DatasetConfig, num_idxs, random_seed):
    np.random.seed(random_seed)

    dataset = get_dataset(ds_config)
    try:
        format_func = getattr(formatters, ds_config.formatter)
    except AttributeError:
        raise ValueError(f"Unable to map dataset ({ds_config.name}) to formatter function")

    if not dataset.get(ds_config.split):
        raise ValueError(f"Invalid split ({ds_config.split}) set for dataset ({ds_config.name})")

    eval_idxs = get_eval_idxs(num_idxs, len(dataset[ds_config.split]))

    return format_func(dataset, eval_idxs, ds_config.split)


def download_hf_dataset(ds_config, ds_path, force=False):
    # dont redownload the dataset unless the user forces it
    if force or not os.path.exists(ds_path):
        print(f"Downloading dataset: ({ds_config.name})")
        dataset = load_dataset(
            ds_config.name,
            ds_config.version,
        )
        dataset.save_to_disk(ds_path)
    else:
        print(f"Dataset ({ds_config.name}) already exists. Set force=True to redownload")
        dataset = load_from_disk(ds_path)

    return dataset


def download_url_dataset(ds_config, ds_path, force=False):
    if force or not os.path.exists(ds_path):
        try:
            dataset = load_dataset(
                "json", 
                data_files={"train":str(ds_config.source)}
            )
            dataset.save_to_disk(ds_path)
        except Exception as e:
            print(f'Error while downloading dataset from URL ({ds_config.name})')
            print(e)
    else:
        print(f"Dataset ({ds_config.name}) already exists. Set force=True to redownload")
        dataset = load_from_disk(ds_path)

    return dataset

def get_dataset(ds_config):
    ensure_directory_exists(DATASETS_DIR)
    dataset_path = os.path.join(DATASETS_DIR, ds_config.name.split('/')[-1])

    if ds_config.source_type == SourceTypes.HUGGINGFACE_HUB:
        dataset = download_hf_dataset(ds_config, dataset_path)

    elif ds_config.source_type == SourceTypes.URL:
        dataset = download_url_dataset(ds_config, dataset_path)
    

    return dataset


def get_eval_idxs(num_idxs, max_idx):
    return np.random.randint(low=0, high=max_idx, size=num_idxs)




if __name__ == "__main__":
    # data = [
    #     ('ms_marco','v1.1'),
    #     ('cnn_dailymail', '3.0.0'),
    #     ('dim/mt_bench_en', None),
    #     ('xsum', None),
    #     ('quac', None),
    #     ('squad_v2', None)
    # ]
    # for ds, version in data:
    #     download_hf_dataset(ds, version)
    ds_config = DatasetConfig(
        name="test_ds",
        source="https://drive.google.com/uc?export=download&id=1uVJbsgPCHFAvH43I6SVvU3Ayo8dh-y_N",
        scenarios=[],
        formatter="",
        source_type=SourceTypes.URL
    )
    download_url_dataset()


