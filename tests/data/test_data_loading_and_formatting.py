from .fixtures import dataset_names, dataset_configs
import pytest
from gpt_eval.config import DatasetConfig, load_validated_config
from gpt_eval.utils import get_dataset_config
from gpt_eval.data.loader import load_formatted_data


@pytest.mark.parametrize(
    "dataset_names, dataset_config", [(dataset_names(), dataset_configs())]
)
def test_can_get_config_from_dataset_name(dataset_names, dataset_config):
    dataset_config = load_validated_config(dataset_config, DatasetConfig, True)
    for name in dataset_names:
        config = get_dataset_config(name, dataset_config)
        assert isinstance(config, DatasetConfig)
        assert config.name == name
