import pathlib
from typing import Optional

from .data_models import (
    DatasetConfig,
    EvaluatedModel,
    EvaluationConfig,
    MetricGroupConfig,
    TaskConfig,
    MetricConfig,
    RunConfig,
)
from .settings import (
    DATASET_CONFIG_PATH,
    DATASETS_DIR,
    EVAL_CONFIG_PATH,
    RUN_CONFIG_PATH,
    REQUEST_RETRY_BACKOFF,
    REQUEST_RETRY_MAX_ATTEMPTS,
    REQUEST_RETRY_WAIT_TIME,
    ApiTypes,
    ModelFamilyTypes,
    TaskTypes,
    SourceTypes,
    IgnoreCacheTypes,
)
from .validator import (
    check_tasks_valid_for_dataset,
    load_and_validate_configs,
    load_validated_config,
)


def get_config_definitions(
    eval_config: Optional[pathlib.Path],
    dataset_config: Optional[pathlib.Path],
    run_config: Optional[pathlib.Path],
):
    return [
        {
            "cls": EvaluationConfig,
            "path": eval_config or EVAL_CONFIG_PATH,
            "is_list": False,
            "key": "eval",
        },
        {
            "cls": DatasetConfig,
            "path": dataset_config or DATASET_CONFIG_PATH,
            "is_list": True,
            "key": "datasets",
        },
        {
            "cls": RunConfig,
            "path": run_config or RUN_CONFIG_PATH,
            "is_list": False,
            "key": "run",
        },
    ]
