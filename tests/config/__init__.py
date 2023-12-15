VALID_SCENARIOS = [
    {
        "name": "Response Quality",
        "id": "rq",
        "score_min": 0,
        "score_max": 10,
        "datasets": ["dim/mt_bench_en"],
        "metrics": [
            {"name": "Accuracy", "desc": "accuracy"},
            {"name": "Coherence", "desc": "choherence"},
        ],
    }
]

VALID_TASKS = [{"name": "QA", "id": "st_qa", "desc": "Question Answering"}]

VALID_RUN_CONFIG = {
    "random_seed": 123,
    "num_evals": 2,
    "max_tokens": 2048,
    "context_char_limit": 1024,
    "temperature": 0.5,
    "judge": {
        "name": "gpt-3.5-turbo",
        "api_key": "test",
        "api_type": "openai",
        "temperature": 0.3,
        "use_proxy": False,
        "proxies": {"http": "http://notarealurl", "https": "http://notarealurl"},
    },
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
    "scenarios": ["rq"],
}
