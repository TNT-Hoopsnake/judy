import pytest
from pydantic import ValidationError
from judy.cli.run import EvalCommandLine
from judy.config import (
    load_validated_config,
    EvaluationConfig,
    DatasetConfig,
    RunConfig,
)
from . import VALID_TASKS, VALID_RUN_CONFIG
from .fixtures import (
    get_param_ids,
    get_param_data,
    VALID_RUN_PARAMS,
    INVALID_RUN_PARAMS,
    VALID_EVAL_PARAMS,
    INVALID_EVAL_PARAMS,
    VALID_DATASET_PARAMS,
    INVALID_DATASET_PARAMS,
)


def test_invalid_dataset_eval_config():
    cli = EvalCommandLine()
    eval_config = EvaluationConfig(
        scenarios=[
            {
                "name": "Response Quality",
                "id": "rq",
                "score_min": 0,
                "score_max": 10,
                "datasets": ["this_doesnt_exist"],
                "metrics": [
                    {"name": "Accuracy", "desc": "accuracy"},
                    {"name": "Coherence", "desc": "choherence"},
                ],
            }
        ],
        tasks=VALID_TASKS,
    )
    with pytest.raises(ValueError):
        cli.collect_evaluations(
            run_config=RunConfig(**VALID_RUN_CONFIG),
            eval_config=eval_config,
            dataset_config_list=[
                DatasetConfig(
                    **{
                        "id": "ms_marco",
                        "name": "MS MARCO",
                        "source": "https://huggingface.co/datasets/ms_marco",
                        "version": "v1.1",
                        "tasks": ["st_qa"],
                        "formatter": "msmarco_formatter",
                    }
                ),
            ],
        )


def test_invalid_task_for_dataset():
    cli = EvalCommandLine()
    eval_config = EvaluationConfig(
        scenarios=[
            {
                "name": "Response Quality",
                "id": "rq",
                "score_min": 0,
                "score_max": 10,
                "datasets": ["ms_marco"],
                "metrics": [
                    {"name": "Accuracy", "desc": "accuracy"},
                    {"name": "Coherence", "desc": "choherence"},
                ],
            }
        ],
        tasks=VALID_TASKS,
    )
    with pytest.raises(ValueError):
        cli.collect_evaluations(
            run_config=RunConfig(**VALID_RUN_CONFIG),
            eval_config=eval_config,
            dataset_config_list=[
                DatasetConfig(
                    **{
                        "id": "ms_marco",
                        "name": "MS MARCO",
                        "source": "https://huggingface.co/datasets/ms_marco",
                        "version": "v1.1",
                        "tasks": ["this_doesnt_exist"],
                        "formatter": "msmarco_formatter",
                    }
                ),
            ],
        )


# Test for validating a run configuration
@pytest.mark.parametrize(
    "config_data", get_param_data(VALID_RUN_PARAMS), ids=get_param_ids(VALID_RUN_PARAMS)
)
def test_valid_run_config(config_data):
    """
    Test that the function load_validated_config correctly validates a valid EvaluationConfig with run parameters.

    Parameters:
    - config_data (dict): A dictionary representing a valid EvaluationConfig.

    Expected Outcome:
    The test should return a validated EvaluationConfig instance.

    """
    validated_config = load_validated_config(config_data, RunConfig)
    assert isinstance(validated_config, RunConfig)


# Test for validating an evaluation configuration
@pytest.mark.parametrize(
    "config_data",
    get_param_data(VALID_EVAL_PARAMS),
    ids=get_param_ids(VALID_EVAL_PARAMS),
)
def test_valid_eval_config(config_data):
    """
    Test that the function load_validated_config correctly validates a valid EvaluationConfig.

    Parameters:
    - config_data (dict): A dictionary representing a valid EvaluationConfig.

    Expected Outcome:
    The test should return a validated EvaluationConfig instance.

    """
    validated_config = load_validated_config(config_data, EvaluationConfig)
    assert isinstance(validated_config, EvaluationConfig)


# Test for validating a list of dataset configurations
@pytest.mark.parametrize(
    "config_data",
    get_param_data(VALID_DATASET_PARAMS),
    ids=get_param_ids(VALID_DATASET_PARAMS),
)
def test_valid_dataset_config(config_data):
    """
    Test that the function load_validated_config correctly validates a list of valid DatasetConfig instances.

    Parameters:
    - config_data (list of dict): A list of dictionaries representing valid DatasetConfig instances.

    Expected Outcome:
    The test should return a list of validated DatasetConfig instances.

    """
    validated_config = load_validated_config(config_data, DatasetConfig)
    for validated in validated_config:
        assert isinstance(validated, DatasetConfig)


# Test for validating an invalid evaluation configuration
@pytest.mark.parametrize(
    "config_data",
    get_param_data(INVALID_EVAL_PARAMS),
    ids=get_param_ids(INVALID_EVAL_PARAMS),
)
def test_invalid_eval_config(config_data):
    """
    Test that the function load_validated_config correctly raises an exception for an invalid EvaluationConfig.

    Parameters:
    - config_data (dict): A dictionary representing an invalid EvaluationConfig.

    Expected Outcome:
    The test should raise a Pydantic ValidationError.

    """
    with pytest.raises(ValidationError):
        load_validated_config(config_data, EvaluationConfig)


# Test for validating an invalid run configuration
@pytest.mark.parametrize(
    "config_data",
    get_param_data(INVALID_RUN_PARAMS),
    ids=get_param_ids(INVALID_RUN_PARAMS),
)
def test_invalid_run_config(config_data):
    """
    Test that the function load_validated_config correctly raises an exception for an invalid RunConfig with run parameters.

    Parameters:
    - config_data (dict): A dictionary representing an invalid RunConfig.

    Expected Outcome:
    The test should raise a Pydantic ValidationError.

    """
    with pytest.raises(ValidationError):
        load_validated_config(config_data, RunConfig)


# Test for validating a list of invalid dataset configurations
@pytest.mark.parametrize(
    "config_data",
    get_param_data(INVALID_DATASET_PARAMS),
    ids=get_param_ids(INVALID_DATASET_PARAMS),
)
def test_invalid_dataset_config(config_data):
    """
    Test that the function load_validated_config correctly raises an exception for a list of invalid DatasetConfig instances.

    Parameters:
    - config_data (list of dict): A list of dictionaries representing invalid DatasetConfig instances.

    Expected Outcome:
    The test should raise a Pydantic ValidationError.

    """
    with pytest.raises(ValidationError):
        load_validated_config(config_data, DatasetConfig)
