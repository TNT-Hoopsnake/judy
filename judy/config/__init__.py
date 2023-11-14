import json
import pathlib
from typing import Optional, List

from .data_models import (
    DatasetConfig,
    EvaluatedModel,
    EvaluationConfig,
    ScenarioConfig,
    TaskConfig,
    MetricConfig,
    RunConfig,
)
from .settings import (
    DATASET_CONFIG_PATH,
    DATASETS_DIR,
    EVAL_CONFIG_PATH,
    RUN_CONFIG_PATH,
    LOG_FILE_PATH,
    REQUEST_RETRY_BACKOFF,
    REQUEST_RETRY_MAX_ATTEMPTS,
    REQUEST_RETRY_WAIT_TIME,
    DEFAULT_OPENAI_API_BASE,
    ApiTypes,
    JudgeModels,
    ModelFamilyTypes,
    TaskTypes,
    SourceTypes,
    IgnoreCacheTypes,
    InputTokenCost,
    OutputTokenCost,
)
from .validator import (
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
            "key": "eval",
        },
        {
            "cls": DatasetConfig,
            "path": dataset_config or DATASET_CONFIG_PATH,
            "key": "datasets",
        },
        {
            "cls": RunConfig,
            "path": run_config or RUN_CONFIG_PATH,
            "key": "run",
        },
    ]


def dump_configs(dir_path: str, configs):
    with open(dir_path / "config.json", "w+") as fn:
        data = {}
        for key, config in configs.items():
            if isinstance(config, list):
                data[key] = [c.model_dump(mode="json") for c in config]
            else:
                data[key] = config.model_dump(mode="json")

        json.dump(data, fn, indent=4)


def get_dataset_config(
    dataset_id: str, ds_config_list: List[DatasetConfig]
) -> DatasetConfig:
    try:
        matching_dataset = next(
            filter(
                lambda dataset: dataset.id
                == dataset_id,  # pylint: disable=cell-var-from-loop
                ds_config_list,
            )
        )
    except StopIteration:
        matching_dataset = None
    if not matching_dataset:
        raise ValueError(
            f"Dataset {dataset_id} is undefined. Create an entry for it in the dataset config"
        )
    return matching_dataset


def get_task_config(task_id: str, eval_config: EvaluationConfig):
    try:
        matching_task = next(
            filter(
                lambda task: task.id == task_id,  # pylint: disable=cell-var-from-loop
                eval_config.tasks,
            )
        )
    except StopIteration:
        matching_task = None
    if not matching_task:
        raise ValueError(
            f"Dataset {task_id} is undefined. Create an entry for it in the dataset config"
        )
    return matching_task


def get_scenario_config(scenario_id: str, eval_config: EvaluationConfig):
    try:
        matching_scenario = next(
            filter(
                lambda scenario: scenario.id
                == scenario_id,  # pylint: disable=cell-var-from-loop
                eval_config.scenarios,
            )
        )
    except StopIteration:
        matching_scenario = None
    if not matching_scenario:
        raise ValueError(
            f"Dataset {scenario_id} is undefined. Create an entry for it in the dataset config"
        )
    return matching_scenario


def get_est_token_cost(
    eval_model: JudgeModels, num_input_tokens: int, num_output_tokens
) -> float:
    match eval_model:
        case JudgeModels.GPT4:
            input_cost = InputTokenCost.GPT4
            output_cost = OutputTokenCost.GPT4
        case JudgeModels.GPT35:
            input_cost = InputTokenCost.GPT35
            output_cost = InputTokenCost.GPT35
        case _:
            input_cost = output_cost = 0

    return round((num_input_tokens * input_cost) + (num_output_tokens * output_cost), 5)
