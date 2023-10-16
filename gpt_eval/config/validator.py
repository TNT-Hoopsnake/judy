from pydantic import TypeAdapter, conlist
import json


def load_json_from_file(filepath):
    with open(filepath, 'r') as fn:
        json_data = json.load(fn)

    return json_data


def get_validated_data(data, cls):
    validated_data = cls(**data)
    return validated_data


def get_validated_list(data, cls):
    validated_list = TypeAdapter(conlist(cls, min_length=1)).validate_python(data)
    return validated_list


def load_validated_config(json_data, validate_cls, is_list=False):
    if is_list:
        validated_data = get_validated_list(json_data, validate_cls)
    else:
        validated_data = get_validated_data(json_data, validate_cls)
    return validated_data


def check_scenarios_valid_for_dataset(eval_config, datasets_config):
    for scenario in eval_config.scenarios:
        for dataset_name in scenario.datasets:
            dataset_config = next((config for config in datasets_config if config.name == dataset_name), None)
            if not dataset_config:
                raise ValueError(f"Dataset '{dataset_name}' is not configured.")
            if scenario.type not in dataset_config.scenarios:
                raise ValueError(f"Scenario type '{scenario.type}' is not valid for dataset '{dataset_name}'.")
            

def load_and_validate_configs(config_definitions):
    configs = {}
    for config_def in config_definitions:
        try:
            json_data = load_json_from_file(config_def['path'])
            validated_data = load_validated_config(json_data, config_def['cls'], config_def['is_list'])
            configs[config_def['key']] = validated_data
        except Exception as e:
            print(f"Error while validating {config_def['key']} config")
            print(e)
            exit(1)

    try:
        check_scenarios_valid_for_dataset(configs['eval'], configs['datasets'])
    except ValueError as e:
        print(e)
        exit(1)

    return configs