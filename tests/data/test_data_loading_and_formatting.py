import pytest
from .fixtures import dataset_ids, dataset_configs
from judy.config import DatasetConfig, load_validated_config, get_dataset_config


@pytest.mark.parametrize(
    "dataset_ids, dataset_config", [(dataset_ids(), dataset_configs())]
)
def test_can_get_config_from_dataset_id(dataset_ids, dataset_config):
    dataset_config = load_validated_config(dataset_config, DatasetConfig)
    for d_id in dataset_ids:
        config = get_dataset_config(d_id, dataset_config)
        assert isinstance(config, DatasetConfig)
        assert config.id == d_id
