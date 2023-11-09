import sys
import logging
import yaml
from pydantic import TypeAdapter, conlist

_logger = logging.getLogger("app")


def load_yaml_from_file(filepath):
    try:
        with open(filepath, "r") as fn:
            data = yaml.safe_load(fn)
    except yaml.YAMLError as e:
        _logger.error("Failed to load YAML data from path: %s - %s", filepath, e)
        sys.exit(1)
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


def load_and_validate_configs(config_definitions):
    configs = {}
    for config_def in config_definitions:
        try:
            config_data = load_yaml_from_file(config_def["path"])
            validated_data = load_validated_config(config_data, config_def["cls"])
            configs[config_def["key"]] = validated_data
        except Exception as e:
            _logger.error("Failed to validate config for %s - %s", config_def["key"], e)
            sys.exit(1)

    return configs
