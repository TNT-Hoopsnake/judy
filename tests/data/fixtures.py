def dataset_configs():
    return [
        {
            "id": "dim/mt_bench_en",
            "source": "https://huggingface.co/datasets/dim/mt_bench_en",
            "version": None,
            "tasks": [{"id": "mt_q", "formatter": "mtbench_formatter"}],
        },
        {
            "id": "ms_marco",
            "source": "https://huggingface.co/datasets/ms_marco",
            "version": "v1.1",
            "tasks": [{"id": "st_qac", "formatter": "msmarco_formatter"}],
        },
        {
            "id": "xsum",
            "source": "https://huggingface.co/datasets/xsum",
            "tasks": [{"id": "summ", "formatter": "xsum_formatter"}],
            "version": None,
        },
        {
            "id": "cnn_dailymail",
            "source": "https://huggingface.co/datasets/cnn_dailymail",
            "version": "3.0.0",
            "tasks": [{"id": "summ", "formatter": "cnn_formatter"}],
        },
        {
            "id": "squad_v2",
            "source": "https://huggingface.co/datasets/squad_v2",
            "version": None,
            "tasks": [{"id": "st_qac", "formatter": "squad_formatter"}],
            "formatter": "squad_formatter",
        },
        {
            "id": "quac",
            "source": "https://huggingface.co/datasets/quac",
            "version": None,
            "tasks": [{"id": "mt_qac", "formatter": "quac_formatter"}],
            "formatter": "quac_formatter",
        },
    ]


def dataset_ids():
    return ["dim/mt_bench_en", "ms_marco", "xsum", "cnn_dailymail", "squad_v2", "quac"]
