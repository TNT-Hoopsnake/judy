from gpt_eval.config import EvaluationConfig, DatasetConfig
from . import VALID_METRIC_GROUPS, VALID_TASKS, VALID_RUN_CONFIG


def valid_eval_dataset_config():
    eval_config = EvaluationConfig(
        metric_groups=VALID_METRIC_GROUPS,
        tasks=VALID_TASKS,
        models=[
            {
                "id": "test-model",
                "api_type": "tgi",
                "api_base": "http://fake.domain",
                "tags": [],
            }
        ],
    )
    dataset_config = [
        DatasetConfig(
            id="fake_summ",
            source="http://fake.dataset",
            tasks=["summ"],
            formatter="fake_formatter",
        )
    ]
    return eval_config, dataset_config


def invalid_task_for_dataset_config():
    # task for given dataset in eval is not valid according to given dataset config
    eval_config = EvaluationConfig(
        metric_groups=VALID_METRIC_GROUPS,
        tasks=[{"id": "summ", "datasets": ["fake_summ"]}],
    )
    dataset_config = [
        DatasetConfig(
            id="fake_summ",
            source="http://fake.dataset",
            tasks=["mt_q"],
            formatter="fake_formatter",
        )
    ]
    return eval_config, dataset_config


VALID_RUN_PARAMS = [
    (
        "define proxies and use them",
        {
            **VALID_RUN_CONFIG,
            "judge": "gpt-3.5-turbo",
            "judge_api_key": None,
            "judge_temperature": 1,
            "use_proxy": True,
            "proxies": {"http": "http://sample-proxy", "https": "http://sample-proxy"},
        },
    ),
    (
        "no proxies and disabled proxy use",
        {
            **VALID_RUN_CONFIG,
            "judge": "gpt-4",
            "judge_api_key": "hf-whatever",
            "judge_temperature": 0.2,
            "use_proxy": False,
        },
    ),
    (
        "no judge api key",
        {
            **VALID_RUN_CONFIG,
            "judge": "gpt-4",
            "judge_temperature": 1.9,
            "use_proxy": False,
        },
    ),
    (
        "define multiple models",
        {
            **VALID_RUN_CONFIG,
            "models": [
                {
                    "id": "flan-t5-small",
                    "api_type": "tgi",
                    "api_base": "http://localhost:8080",
                },
                {
                    "id": "flan-example",
                    "api_type": "openai",
                    "api_base": "http://not.real",
                },
            ],
        },
    ),
    (
        "define single model",
        {
            **VALID_RUN_CONFIG,
            "models": [
                {
                    "id": "flan-t5-small",
                    "api_type": "tgi",
                    "api_base": "http://localhost:8080",
                    "temperature": 0.5,
                    "max_tokens": 300,
                    "context_char_limit": 10,
                }
            ],
        },
    ),
]


INVALID_RUN_PARAMS = [
    (
        "proxies must exist when use_proxy is true",
        {
            **{k: v for k, v in VALID_RUN_CONFIG.items() if k != "proxies"},
            "judge": "gpt-3.5-turbo",
            "judge_api_key": None,
            "judge_temperature": 1,
            "use_proxy": True,
        },
    ),
    (
        "judge temperature must be between 0 and 2",
        {
            **VALID_RUN_CONFIG,
            "judge": "gpt-4",
            "judge_api_key": "hf-whatever",
            "judge_temperature": 3.0,
            "use_proxy": False,
        },
    ),
    (
        "use_proxy field must exist",
        {
            **{k: v for k, v in VALID_RUN_CONFIG.items() if k != "use_proxy"},
            "judge": "gpt-4",
            "judge_temperature": 1.9,
        },
    ),
    (
        "max tokens is a negative value",
        {
            **VALID_RUN_CONFIG,
            "max_tokens": -1,
        },
    ),
    (
        "missing api_base",
        {
            **VALID_RUN_CONFIG,
            "models": [
                {
                    "id": "flan-t5-small",
                    "api_type": "tgi",
                },
            ],
        },
    ),
    (
        "evaluated models is empty",
        {
            **VALID_RUN_CONFIG,
            "models": [],
        },
    ),
]


VALID_EVAL_PARAMS = [
    (
        "define multiple tasks",
        {
            "metric_groups": VALID_METRIC_GROUPS,
            "tasks": VALID_TASKS,
        },
    ),
]


INVALID_EVAL_PARAMS = [
    (
        "no datasets for task",
        {
            "metric_groups": VALID_METRIC_GROUPS,
            "tasks": [
                {
                    "name": "QA",
                    "id": "st_qa",
                    "desc": "Question Answering",
                    "datasets": [],
                }
            ],
        },
    ),
    (
        "no tasks for metric group",
        {
            "metric_groups": [
                {
                    "name": " Response Quality",
                    "id": "rq",
                    "min": 0,
                    "max": 10,
                    "tasks": [],
                    "metrics": [
                        {"name": "Accuracy", "desc": "accuracy"},
                        {"name": "Coherence", "desc": "choherence"},
                    ],
                }
            ],
            "tasks": VALID_TASKS,
        },
    ),
]


VALID_DATASET_PARAMS = [
    (
        "no version specified",
        [
            {
                "id": "dim/mt_bench_en",
                "name": "MT BENCH",
                "source": "https://huggingface.co/datasets/dim/mt_bench_en",
                "version": None,
                "tasks": ["mt_q"],
                "formatter": "mtbench_formatter",
            },
            {
                "id": "ms_marco",
                "name": "MS MARCO",
                "source": "https://huggingface.co/datasets/ms_marco",
                "version": "v1.1",
                "tasks": ["st_qac"],
                "formatter": "msmarco_formatter",
            },
        ],
    ),
    (
        "versions defined",
        [
            {
                "id": "dim/mt_bench_en",
                "name": "MT BENCH",
                "source": "https://huggingface.co/datasets/dim/mt_bench_en",
                "tasks": ["mt_q"],
                "formatter": "mtbench_formatter",
            },
            {
                "id": "ms_marco",
                "name": "MS MARCO",
                "source": "https://huggingface.co/datasets/ms_marco",
                "version": "v1.1",
                "tasks": ["st_qac"],
                "formatter": "msmarco_formatter",
            },
        ],
    ),
]


INVALID_DATASET_PARAMS = [
    (
        "dataset id is missing",
        [
            {
                "source": "https://huggingface.co/datasets/dim/mt_bench_en",
                "version": None,
                "tasks": ["mt_q"],
                "formatter": "mtbench_formatter",
            },
            {
                "id": "ms_marco",
                "source": "https://huggingface.co/datasets/ms_marco",
                "version": "v1.1",
                "tasks": ["st_qac"],
                "formatter": "msmarco_formatter",
            },
        ],
    ),
    ("must contain at least one dataset config", []),
    (
        "dataset tasks must contain at least one item",
        [
            {
                "id": "mt_bench",
                "name": "MT BENCH",
                "source": "https://huggingface.co/datasets/dim/mt_bench_en",
                "version": None,
                "tasks": [],
                "formatter": "mtbench_formatter",
            },
            {
                "id": "ms_marco",
                "name": "MS MARCO",
                "source": "https://huggingface.co/datasets/ms_marco",
                "version": "v1.1",
                "tasks": ["st_qac"],
                "formatter": "msmarco_formatter",
            },
        ],
    ),
]


def get_param_ids(values):
    for value in values:
        yield value[0]


def get_param_data(values):
    for value in values:
        yield value[1]
