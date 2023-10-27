VALID_METRIC_GROUPS = [
    {
        "name": " Response Quality",
        "id": "rq",
        "min": 0,
        "max": 10,
        "tasks": ["mt_q", "mt_qac", "st_qac", "summ", "st_q", "st_qa"],
        "metrics": [
            {"name": "Accuracy", "desc": "accuracy"},
            {"name": "Coherence", "desc": "choherence"},
        ],
    }
]

VALID_TASKS = [
    {
        "name": "QA",
        "id": "st_qa",
        "desc": "Question Answering",
        "datasets": ["riddle_sense"],
    }
]

VALID_RUN_CONFIG = {
    "judge": "gpt-3.5-turbo",
    "judge_temperature": 0.3,
    "use_proxy": False,
    "proxies": {"http": "http://notarealurl", "https": "http://notarealurl"},
    "random_seed": 123,
    "num_evals": 2,
    "max_tokens": 2048,
    "context_char_limit": 1024,
    "temperature": 0.5,
    "models": ["test-model"],
    "tasks": ["st_q"],
    "metrics": ["accuracy"],
}
