from .fixtures import (
    valid_system_configs,
    invalid_system_configs,
    valid_eval_configs,
    invalid_eval_configs,
    valid_dataset_configs,
    invalid_dataset_configs,
    valid_eval_dataset_config,
    invalid_eval_dataset_config,
)
import pytest
from gpt_eval.config import (
    load_validated_config,
    EvaluationConfig,
    DatasetConfig,
    check_scenarios_valid_for_dataset,
)
from pydantic import ValidationError


# Test for checking if scenarios are valid for a dataset
@pytest.mark.parametrize("eval_config, dataset_config", [valid_eval_dataset_config()])
def test_datasets_valid_for_scenario(eval_config, dataset_config):
    """
    Test that the function check_scenarios_valid_for_dataset correctly validates scenarios for a given dataset configuration.

    Parameters:
    - eval_config (EvaluationConfig): A valid evaluation configuration.
    - dataset_config (list of DatasetConfig): A list of valid dataset configurations.

    Expected Outcome:
    The test should pass without raising any exceptions.

    """
    check_scenarios_valid_for_dataset(eval_config, dataset_config)


# Test for checking if scenarios are invalid for a dataset
@pytest.mark.parametrize("eval_config, dataset_config", [invalid_eval_dataset_config()])
def test_datasets_valid_for_scenario(eval_config, dataset_config):
    """
    Test that the function check_scenarios_valid_for_dataset correctly detects invalid scenarios for a given dataset configuration.

    Parameters:
    - eval_config (EvaluationConfig): An invalid evaluation configuration.
    - dataset_config (list of DatasetConfig): A list of dataset configurations with invalid scenarios.

    Expected Outcome:
    The test should raise a ValueError indicating that scenarios are invalid.

    """
    with pytest.raises(ValueError):
        check_scenarios_valid_for_dataset(eval_config, dataset_config)


# Test for validating a system configuration
@pytest.mark.parametrize("config_data", valid_system_configs())
def test_valid_system_config(config_data):
    """
    Test that the function load_validated_config correctly validates a valid EvaluationConfig with system parameters.

    Parameters:
    - config_data (dict): A dictionary representing a valid EvaluationConfig.

    Expected Outcome:
    The test should return a validated EvaluationConfig instance.

    """
    validated_config = load_validated_config(config_data, EvaluationConfig)
    assert isinstance(validated_config, EvaluationConfig)


# Test for validating an evaluation configuration
@pytest.mark.parametrize("config_data", valid_eval_configs())
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
@pytest.mark.parametrize("config_data", valid_dataset_configs())
def test_valid_dataset_config(config_data):
    """
    Test that the function load_validated_config correctly validates a list of valid DatasetConfig instances.

    Parameters:
    - config_data (list of dict): A list of dictionaries representing valid DatasetConfig instances.

    Expected Outcome:
    The test should return a list of validated DatasetConfig instances.

    """
    validated_config = load_validated_config(config_data, DatasetConfig, True)
    for validated in validated_config:
        assert isinstance(validated, DatasetConfig)


# Test for validating an invalid evaluation configuration
@pytest.mark.parametrize("config_data", invalid_eval_configs())
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


# Test for validating an invalid system configuration
@pytest.mark.parametrize("config_data", invalid_system_configs())
def test_invalid_system_config(config_data):
    """
    Test that the function load_validated_config correctly raises an exception for an invalid EvaluationConfig with system parameters.

    Parameters:
    - config_data (dict): A dictionary representing an invalid EvaluationConfig.

    Expected Outcome:
    The test should raise a Pydantic ValidationError.

    """
    with pytest.raises(ValidationError):
        load_validated_config(config_data, EvaluationConfig)


# Test for validating a list of invalid dataset configurations
@pytest.mark.parametrize("config_data", invalid_dataset_configs())
def test_invalid_dataset_config(config_data):
    """
    Test that the function load_validated_config correctly raises an exception for a list of invalid DatasetConfig instances.

    Parameters:
    - config_data (list of dict): A list of dictionaries representing invalid DatasetConfig instances.

    Expected Outcome:
    The test should raise a Pydantic ValidationError.

    """
    with pytest.raises(ValidationError):
        load_validated_config(config_data, DatasetConfig, True)
