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


def load_validated_config(json_data, validate_cls, is_list=False):
    if is_list:
        validated_data = get_validated_list(json_data, validate_cls)
    else:
        validated_data = get_validated_data(json_data, validate_cls)
    return validated_data


def load_and_validate_configs(config_definitions):
    configs = {}
    for config_def in config_definitions:
        try:
            config_data = load_yaml_from_file(config_def["path"])
            validated_data = load_validated_config(
                config_data, config_def["cls"], config_def["is_list"]
            )
            configs[config_def["key"]] = validated_data
        except Exception as e:
            print(f"Error while validating {config_def['key']} config")
            print(config_data)
            print(e)
            sys.exit(1)

    return configs
