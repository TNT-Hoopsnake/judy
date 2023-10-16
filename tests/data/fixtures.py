def dataset_configs():
    return [
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
        {
            "name":"xsum",
            "source": "https://huggingface.co/datasets/xsum",
            "scenarios": ["summ"],
            "formatter": "xsum_formatter",
            "version": None
        },
        {
            "name":"cnn_dailymail",
            "source": "https://huggingface.co/datasets/cnn_dailymail",
            "version": "3.0.0",
            "scenarios": ["summ"],
            "formatter": "cnn_formatter"
        },
        {
            "name":"squad_v2",
            "source": "https://huggingface.co/datasets/squad_v2",
            "version": None,
            "scenarios": ["st_qac"],
            "formatter": "squad_formatter"
        },
        {
            "name":"quac",
            "source": "https://huggingface.co/datasets/quac",
            "version": None,
            "scenarios": ["mt_qac"],
            "formatter": "quac_formatter"
        }
    ]


def dataset_names():
    return [
        'dim/mt_bench_en',
        'ms_marco',
        'xsum',
        'cnn_dailymail',
        'squad_v2',
        'quac'
    ]
