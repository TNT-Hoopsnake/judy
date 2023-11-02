import pytest
from pydantic import ValidationError
from gpt_eval.config import (
    load_validated_config,
    EvaluationConfig,
    DatasetConfig,
    RunConfig,
    check_tasks_valid_for_dataset,
)
from .fixtures import (
    get_param_ids,
    get_param_data,
    VALID_RUN_PARAMS,
    INVALID_RUN_PARAMS,
    VALID_EVAL_PARAMS,
    INVALID_EVAL_PARAMS,
    VALID_DATASET_PARAMS,
    INVALID_DATASET_PARAMS,
    valid_eval_dataset_config,
    invalid_task_for_dataset_config,
)


# Test for checking if tasks are valid for a dataset
@pytest.mark.parametrize("eval_config, dataset_config", [valid_eval_dataset_config()])
def test_datasets_valid_for_task(eval_config, dataset_config):
    """
    Test that the function check_tasks_valid_for_dataset correctly validates tasks for a given dataset configuration.

    Parameters:
    - eval_config (EvaluationConfig): A valid evaluation configuration.
    - dataset_config (list of DatasetConfig): A list of valid dataset configurations.

    Expected Outcome:
    The test should pass without raising any exceptions.

    """
    check_tasks_valid_for_dataset(eval_config, dataset_config)


# Test for checking if tasks are invalid for a dataset
@pytest.mark.parametrize(
    "eval_config, dataset_config", [invalid_task_for_dataset_config()]
)
def test_datasets_invalid_for_task(eval_config, dataset_config):
    """
    Test that the function check_tasks_valid_for_dataset correctly detects invalid tasks for a given dataset configuration.

    Parameters:
    - eval_config (EvaluationConfig): An invalid evaluation configuration.
    - dataset_config (list of DatasetConfig): A list of dataset configurations with invalid tasks.

    Expected Outcome:
    The test should raise a ValueError indicating that tasks are invalid.

    """
    with pytest.raises(ValueError):
        check_tasks_valid_for_dataset(eval_config, dataset_config)


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
