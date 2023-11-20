import sys
import yaml
from pydantic import TypeAdapter, conlist
from judy.config.logging import logger as log


def load_yaml_from_file(filepath):
    """
    Attempts to load YAML data from the specified file. If the loading
    fails due to a YAML error, an error is logged, and the program exits.

    Args:
        filepath (str): Path to the YAML file.

    Returns:
        dict: Loaded YAML data.
    """

    try:
        with open(filepath, "r") as fn:
            data = yaml.safe_load(fn)
    except yaml.YAMLError as e:
        log.error("Failed to load YAML data from path: %s - %s", filepath, e)
        sys.exit(1)
    return data


def get_validated_data(data, cls):
    """
    Takes raw data and a pydantic model class definition and returns an instance of
    the class after validating the data against the class attributes.

    Args:
        data (dict): Data to be validated.
        cls (Type): Class to validate against.

    Returns:
        Any: Validated instance of the specified pydantic model class.
    """
    validated_data = cls(**data)
    return validated_data


def get_validated_list(data, cls):
    """
    Takes a list of raw data and a pydantic model class definition and returns a list of
    instances of the class after validating each element in the list against the pydantic class attributes.

    Args:
        data (List[dict]): List of data to be validated.
        cls (Type): Class to validate against.

    Returns:
        List[Any]: List of validated instances of the specified pydantic model class.


    """
    validated_list = TypeAdapter(conlist(cls, min_length=1)).validate_python(data)
    return validated_list


def load_validated_config(config_data, validate_cls):
    """
    Takes configuration data and a pydantic class definition and returns validated
    configuration data, either as an instance of the specified class or as a list of instances,
    depending on the type of the original data.

    Args:
        config_data (Union[dict, List[dict]]): Configuration data to be validated.
        validate_cls (Type): Class to validate against.

    Returns:
        Any: Validated configuration data.


    """
    if isinstance(config_data, list):
        validated_data = get_validated_list(config_data, validate_cls)
    else:
        validated_data = get_validated_data(config_data, validate_cls)
    return validated_data


def load_and_validate_configs(config_definitions):
    """
    Iterates through a list of configuration definitions, loads the YAML data
    from the specified paths, and validates the data against the provided class definitions.
    The validated data is stored in a dictionary with keys corresponding to the configuration keys.
    If any validation fails, an error is logged, and the program exits.

    Args:
        config_definitions (List[dict]): List of configuration definitions.

    Returns:
        dict: Dictionary containing validated configuration data.
    """
    configs = {}
    for config_def in config_definitions:
        try:
            config_data = load_yaml_from_file(config_def["path"])
            validated_data = load_validated_config(config_data, config_def["cls"])
            configs[config_def["key"]] = validated_data
        except Exception as e:
            log.error("Failed to validate config for %s - %s", config_def["key"], e)
            sys.exit(1)

    return configs
