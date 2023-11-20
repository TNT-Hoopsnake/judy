import os

import numpy as np
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from judy.config.settings import TaskTypes
from judy.utils import Retry
from judy.dataset import formatters
from judy.config import DATASETS_DIR, DatasetConfig, SourceTypes
from judy.utils import ensure_directory_exists
from judy.config.logging import logger as log


def load_formatted_data(
    ds_config: DatasetConfig,
    num_idxs: int,
    random_seed: int,
    task_type: TaskTypes,
    ignore_cache: bool = False,
):
    """
    Loads a dataset based on the provided configuration, formats the data
    using the specified formatter class, and returns the formatted data.

    Args:
        ds_config (DatasetConfig): Configuration for the dataset.
        num_idxs (int): Number of indices to evaluate.
        random_seed (int): Seed for the random number generator.
        task_type (TaskTypes): ID of the task that will use the formatted data
        ignore_cache (bool, optional): Flag to ignore the cache and force data reload.

    Returns:
        dict: Formatted data.

    """

    np.random.seed(random_seed)

    dataset = get_dataset(ds_config, ignore_cache)
    if isinstance(dataset, DatasetDict):
        dataset = dataset.get(ds_config.split)
        if not dataset:
            raise ValueError(
                f"Invalid split ({ds_config.split}) set for dataset ({ds_config.id})"
            )
    try:
        dataset_task = next(
            filter(
                lambda task: task.id == task_type,  # pylint: disable=cell-var-from-loop
                ds_config.tasks,
            )
        )
        format_class = getattr(formatters, dataset_task.formatter)
    except AttributeError as exc:
        raise ValueError(
            f"Unable to map dataset ({ds_config.id}) to formatter class"
        ) from exc

    eval_idxs = get_eval_idxs(num_idxs, len(dataset))

    return format_class(dataset, eval_idxs).format()


@Retry
def get_dataset(
    ds_config: DatasetConfig, ignore_cache: bool = False
) -> Dataset | DatasetDict:
    """
    Retrieves a dataset based on the provided configuration. It checks
    the cache for the dataset; if not present or ignoring the cache, it downloads the
    dataset and saves it to the relevant directory.

    Args:
        ds_config (DatasetConfig): Configuration for the dataset.
        ignore_cache (bool, optional): Flag to ignore the cache and force data reload.

    Returns:
        Dataset | DatasetDict: Loaded dataset.


    """
    ensure_directory_exists(DATASETS_DIR)
    ds_path = os.path.join(DATASETS_DIR, ds_config.id.split("/")[-1])
    if ignore_cache or not os.path.exists(ds_path):
        try:
            match ds_config.source_type:
                case SourceTypes.HUGGINGFACE_HUB:
                    dataset = load_dataset(
                        ds_config.id,
                        ds_config.version,
                        split=ds_config.split,
                        streaming=False,
                    )
                case SourceTypes.URL:
                    dataset = load_dataset(
                        "json",
                        data_files={ds_config.split: str(ds_config.source)},
                        split=ds_config.split,
                    )
                case _:
                    raise ValueError(
                        f"Invalid dataset source type provided: {ds_config.source_type:}"
                    )

            dataset.save_to_disk(ds_path)
        except Exception as e:
            log.error(e)
            raise FileNotFoundError(
                f"Error while downloading dataset from URL ({ds_config.id})"
            ) from e
    else:
        log.info("Dataset (%s) loaded from path (%s)", ds_config.id, ds_path)
        dataset = load_from_disk(ds_path)
    return dataset


def get_eval_idxs(num_idxs: int, max_idx: int):
    """
    Generates a specified number of random indices within a specified range.

    Args:
        num_idxs (int): Number of indices to generate.
        max_idx (int): Maximum index value.

    Returns:
        np.ndarray: Array of randomly generated indices.

    """
    return np.random.randint(low=0, high=max_idx, size=num_idxs)
