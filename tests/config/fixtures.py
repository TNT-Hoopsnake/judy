from gpt_eval.config import (
    EvaluationConfig,
    DatasetConfig
)


def valid_eval_dataset_config():
    eval_config = EvaluationConfig(
        random_seed=10,
        num_evals=1,
        scenarios=[{"type":"summ","datasets":["fake_summ"]}],
        max_tokens=10,
        context_char_limit=10,
        temperature=1,
        evaluated_models=[{"name":"test-model","api_type":"tgi","api_base":"http://fake.domain"}]
    )
    dataset_config = [DatasetConfig(
        name="fake_summ",
        source="http://fake.dataset",
        scenarios=["summ"],
        formatter="fake_formatter"
    )]
    return eval_config, dataset_config


def invalid_eval_dataset_config():
    # scenario for given dataset in eval is not valid according to given dataset config
    eval_config = EvaluationConfig(
        random_seed=10,
        num_evals=1,
        scenarios=[{"type":"summ","datasets":["fake_summ"]}],
        max_tokens=10,
        context_char_limit=10,
        temperature=1,
        evaluated_models=[{"name":"test-model","api_type":"tgi","api_base":"http://fake.domain"}]
    )
    dataset_config = [DatasetConfig(
        name="fake_summ",
        source="http://fake.dataset",
        scenarios=["mt_q"],
        formatter="fake_formatter"
    )]
    return eval_config, dataset_config

def valid_system_configs():
    system_configs = [
        {
            "judge":"gpt-3.5-turbo",
            "judge_api_key":None,
            "judge_temperature": 1,
            "use_proxy": True,
            "proxies": {
                "http":"http://sample-proxy",
                "https":"http://sample-proxy"
            }
        },
        {
            "judge":"gpt-4",
            "judge_api_key":"hf-whatever",
            "judge_temperature": 0.2,
            "use_proxy": False,
        },
        {
            "judge":"gpt-4",
            "judge_temperature": 1.9,
            "use_proxy": False,
        },
    ]
    for config in system_configs:
        yield config


def invalid_system_configs():
    system_configs = [
        # proxies must exist when use_proxy is true
        {
            "judge":"gpt-3.5-turbo",
            "judge_api_key":None,
            "judge_temperature": 1,
            "use_proxy": True,
        },
        # judge temperature must be between 0 and 2
        {
            "judge":"gpt-4",
            "judge_api_key":"hf-whatever",
            "judge_temperature": 3.0,
            "use_proxy": False,
        },
        # use_proxy field must exist
        {
            "judge":"gpt-4",
            "judge_temperature": 1.9,
        },
    ]
    for config in system_configs:
        yield config

def valid_eval_configs():
    eval_configs = [
        {
            "random_seed":123,
            "num_evals":1,
            "scenarios": [
                {
                    "type":"summ",
                    "datasets":["xsum", "cnn_dailymail"]
                },
            ],
            "max_tokens":300,
            "context_char_limit":1000,
            "temperature": 1,
            "evaluated_models": [
                {
                    "name":"flan-t5-small",
                    "api_type":"tgi",
                    "api_base":"http://localhost:8080",
                },
                {
                    "name":"flan-example",
                    "api_type":"openai",
                    "api_base":"http://not.real",
                },
            ]
        },
        {
            "random_seed":None,
            "num_evals":1,
            "scenarios": [
                {
                    "type":"summ",
                    "datasets":["xsum", "cnn_dailymail"]
                },
            ],
            "max_tokens":500,
            "context_char_limit":1000,
            "temperature": 1,
            "evaluated_models": [
                {
                    "name":"flan-t5-small",
                    "api_type":"tgi",
                    "api_base":"http://localhost:8080",
                    "temperature":0.5,
                    "max_tokens":300,
                    "context_char_limit":10
                }
            ]
        }
    ]
    for config in eval_configs:
        yield config

def invalid_eval_configs():
    eval_configs = [
        # missing api_base
        {
            "random_seed":123,
            "num_evals":1,
            "scenarios": [
                {
                    "type":"summ",
                    "datasets":["xsum", "cnn_dailymail"]
                },
            ],
            "max_tokens":300,
            "context_char_limit":1000,
            "temperature": 1,
            "evaluated_models": [
                {
                    "name":"flan-t5-small",
                    "api_type":"tgi",
                },
            ]
        },
        # max tokens is a negative value
        {
            "random_seed":None,
            "num_evals":1,
            "scenarios": [
                {
                    "type":"summ",
                    "datasets":["xsum", "cnn_dailymail"]
                },
            ],
            "max_tokens":-1,
            "context_char_limit":1000,
            "temperature": 1,
            "evaluated_models": [
                {
                    "name":"flan-t5-small",
                    "api_type":"tgi",
                    "api_base":"http://localhost:8080",
                    "temperature":0.5,
                    "max_tokens":300,
                    "context_char_limit":10
                }
            ]
        },
        # evaluated models is empty
        {
            "random_seed":None,
            "num_evals":1,
            "scenarios": [
                {
                    "type":"summ",
                    "datasets":["xsum", "cnn_dailymail"]
                },
            ],
            "max_tokens":100,
            "context_char_limit":1000,
            "temperature": 1,
            "evaluated_models": []
        }
    ]
    for config in eval_configs:
        yield config

def valid_dataset_configs():
    dataset_configs = [
        [
            {
                "name":"dim/mt_bench_en",
                "source":"https://huggingface.co/datasets/dim/mt_bench_en",
                "version": None,
                "scenarios": ["mt_q"],
                "formatter": "mtbench_formatter"
            },
            {
                "name":"ms_marco",
                "source": "https://huggingface.co/datasets/ms_marco",
                "version": "v1.1",
                "scenarios": ["st_qac"],
                "formatter": "msmarco_formatter"
            },
        ],
        [
            {
                "name":"dim/mt_bench_en",
                "source":"https://huggingface.co/datasets/dim/mt_bench_en",
                "scenarios": ["mt_q"],
                "formatter": "mtbench_formatter"
            },
            {
                "name":"ms_marco",
                "source": "https://huggingface.co/datasets/ms_marco",
                "version": "v1.1",
                "scenarios": ["st_qac"],
                "formatter": "msmarco_formatter"
            },
        ]
    ]
    for config in dataset_configs:
        yield config


def invalid_dataset_configs():
    dataset_configs = [
        # dataset name is missing
        [
            {
                "source":"https://huggingface.co/datasets/dim/mt_bench_en",
                "version": None,
                "scenarios": ["mt_q"],
                "formatter": "mtbench_formatter"
            },
            {
                "name":"ms_marco",
                "source": "https://huggingface.co/datasets/ms_marco",
                "version": "v1.1",
                "scenarios": ["st_qac"],
                "formatter": "msmarco_formatter"
            },
        ],
        # must contain at least one dataset config
        [],
        # dataset scenarios must contain at least one item
        [
            {
                "name":"mt_bench",
                "source":"https://huggingface.co/datasets/dim/mt_bench_en",
                "version": None,
                "scenarios": [],
                "formatter": "mtbench_formatter"
            },
            {
                "name":"ms_marco",
                "source": "https://huggingface.co/datasets/ms_marco",
                "version": "v1.1",
                "scenarios": ["st_qac"],
                "formatter": "msmarco_formatter"
            },
        ],
    ]
    for config in dataset_configs:
        yield config
