import sys
import yaml
from pydantic import TypeAdapter, conlist


def load_yaml_from_file(filepath):
    try:
        with open(filepath, "r") as fn:
            data = yaml.safe_load(fn)
    except yaml.YAMLError as e:
        print(f"Error loading YAML data from {filepath} - {e}")
        return []
    return data


def get_validated_data(data, cls):
    validated_data = cls(**data)
    return validated_data


def get_validated_list(data, cls):
    validated_list = TypeAdapter(conlist(cls, min_length=1)).validate_python(data)
    return validated_list


def load_validated_config(config_data, validate_cls):
    if isinstance(config_data, list):
        validated_data = get_validated_list(config_data, validate_cls)
    else:
        validated_data = get_validated_data(config_data, validate_cls)
    return validated_data


def check_tasks_valid_for_dataset(eval_config, datasets_config):
    for task in eval_config.tasks:
        for dataset_name in task.datasets:
            dataset_config = next(
                (config for config in datasets_config if config.id == dataset_name),
                None,
            )
            if not dataset_config:
                raise ValueError(f"Dataset '{dataset_name}' is not configured.")
            if task.id not in dataset_config.tasks:
                raise ValueError(
                    f"Task type '{task.id}' is not valid for dataset '{dataset_name}'."
                )


def load_and_validate_configs(config_definitions):
    configs = {}
    for config_def in config_definitions:
        try:
            config_data = load_yaml_from_file(config_def["path"])
            validated_data = load_validated_config(config_data, config_def["cls"])
            configs[config_def["key"]] = validated_data
        except Exception as e:
            print(f"Error while validating {config_def['key']} config")
            print(config_data)
            print(e)
            sys.exit(1)

    try:
        check_tasks_valid_for_dataset(configs["eval"], configs["datasets"])
    except ValueError as e:
        print(e)
        sys.exit(1)

    return configs
