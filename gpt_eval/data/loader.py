import os
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from gpt_eval.utils import Retry
from gpt_eval.data import formatters
from gpt_eval.config import DATASETS_DIR, DatasetConfig, SourceTypes
from gpt_eval.utils import ensure_directory_exists


def load_formatted_data(
    ds_config: DatasetConfig, num_idxs: int, random_seed: int, force: bool = False
):
    np.random.seed(random_seed)

    dataset = get_dataset(ds_config, force)
    if isinstance(dataset, DatasetDict):
        dataset = dataset.get(ds_config.split)
        if not dataset.get(ds_config.split):
            raise ValueError(
                f"Invalid split ({ds_config.split}) set for dataset ({ds_config.name})"
            )
    try:
        format_func = getattr(formatters, ds_config.formatter)
    except AttributeError as exc:
        raise ValueError(
            f"Unable to map dataset ({ds_config.name}) to formatter function"
        ) from exc
    eval_idxs = get_eval_idxs(num_idxs, len(dataset))

    return format_func(dataset, eval_idxs, ds_config.split)


@Retry
def get_dataset(ds_config: DatasetConfig, force: bool = False) -> Dataset | DatasetDict:
    ensure_directory_exists(DATASETS_DIR)
    ds_path = os.path.join(DATASETS_DIR, ds_config.name.split("/")[-1])
    if force or not os.path.exists(ds_path):
        try:
            if ds_config.source_type == SourceTypes.HUGGINGFACE_HUB:
                dataset = load_dataset(
                    ds_config.name,
                    ds_config.version,
                    split=ds_config.split,
                    streaming=False,
                )
            elif ds_config.source_type == SourceTypes.URL:
                dataset = load_dataset(
                    "json",
                    data_files={ds_config.split: str(ds_config.source)},
                    split=ds_config.split,
                )
            dataset.save_to_disk(ds_path)
        except Exception as e:
            raise FileNotFoundError(
                f"Error while downloading dataset from URL ({ds_config.name})"
            ) from e
    else:
        print(
            f"Dataset ({ds_config.name}) already exists. Set force=True to redownload"
        )
        dataset = load_from_disk(ds_path)
    return dataset


def get_eval_idxs(num_idxs: int, max_idx: int):
    return np.random.randint(low=0, high=max_idx, size=num_idxs)
