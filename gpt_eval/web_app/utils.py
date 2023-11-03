import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from gpt_eval.config import (
    EvaluationConfig,
    DatasetConfig,
    RunConfig,
    load_validated_config,
)
from gpt_eval.utils import matches_tag

DATA_DIR = "./results"

CONFIG_CLASS_MAP = {
    "eval": EvaluationConfig,
    "datasets": DatasetConfig,
    "run": RunConfig,
}


def load_configs(config_path):
    configs = {}
    with open(config_path, "r") as fn:
        data = json.load(fn)

    for key, config in data.items():
        config_cls = CONFIG_CLASS_MAP.get(key)
        if not config_cls:
            raise ValueError(f"Config class could not be retrieved using key: {key}")

        config_model = load_validated_config(config, config_cls)
        configs[key] = config_model

    return configs


def gather_data_index():
    index = {
        "datasets":{},
        "tasks":{},
        "scenarios":{},
        "models":{}
    }
    for run_name in os.listdir(DATA_DIR):
        run_path = os.path.join(DATA_DIR, run_name)
        config = load_configs(os.path.join(run_path, "config.json"))

        for task in config["eval"].tasks:
            index["tasks"][task.id] = task

        for dataset in config["datasets"]:
            index["datasets"][dataset.id] = dataset

        for scenario in config["eval"].scenarios:
            index["scenarios"][scenario.id] = scenario

        for model in config["run"].models:
            index["models"][model.id] = model

        return index



def load_all_data():
    all_data_list = {}
    data_index = gather_data_index()
    for run_name in os.listdir(DATA_DIR):
        run_path = os.path.join(DATA_DIR, run_name)

        config = load_configs(os.path.join(run_path, "config.json"))
        with open(os.path.join(run_path, "metadata.json"), "r") as fn:
            metadata = json.load(fn)

        all_data_list[run_name] = {
            "config": config,
            "metadata": metadata,
            "data": {},
            "models_used": [],
            "tasks_used": [],
            "datasets_used": [],
        }
        run_scenarios = [data_index["scenarios"][id] for id in config["run"].scenarios]
        run_dataset_ids = set()
        run_task_ids = set()
        for scenario in run_scenarios:
            run_dataset_ids = run_dataset_ids.union(set(scenario.datasets))
            for dataset_id in scenario.datasets:
                dataset = data_index['datasets'][dataset_id]
                run_task_ids = run_task_ids.union(set(dataset.tasks))

        for task in config["eval"].tasks:
            if matches_tag(task, metadata["task_tags"]) and task.id in run_task_ids:
                all_data_list[run_name]["tasks_used"].append(task)

        for dataset in config["datasets"]:
            if matches_tag(dataset, metadata["dataset_tags"]) and dataset.id in run_dataset_ids:
                all_data_list[run_name]["datasets_used"].append(dataset)

        for model in config["run"].models:
            model_path = os.path.join(run_path, model.id)

            # ensure the model was included in this run
            if matches_tag(model, metadata["model_tags"]) and os.path.exists(
                model_path
            ):
                all_data_list[run_name]["models_used"].append(model)

                all_data_list[run_name]["data"][model.id] = {}

                for dataset_name in os.listdir(model_path):
                    res_path = os.path.join(model_path, dataset_name)
                    dataset_name = dataset_name.replace("-results.json", "")

                    # ensure the dataset was included in this run
                    if os.path.exists(res_path):
                        with open(res_path, "r") as fn:
                            res_data = json.load(fn)

                        all_data_list[run_name]["data"][model.id][
                            dataset_name
                        ] = res_data

    return all_data_list


# todo - implement local timestamps
def get_readable_timestamp(timestamp):
    timestamp_utc = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
    return timestamp_utc.strftime("%Y-%m-%d %H:%M:%S")


def get_grouped_df(df, group_by_field):
    grouped = df.groupby(group_by_field)
    grouped_data = {}

    for model, data in grouped:
        pivoted_data = pd.pivot_table(
            data, index="model", columns="metric", values="score", aggfunc="mean"
        )
        pivoted_data = pivoted_data.round(2).replace(np.nan, "-")

        grouped_data[model] = pivoted_data

    return grouped_data




def format_data(all_data):
    data_list = []

    for run_name, run_data in all_data.items():
        for model_name, model_data in run_data["data"].items():
            for dataset_name, results in model_data.items():
                for result in results:
                    task_id = result['task_id']
                    scenario_id = result['scenario_id']
                    for metric_scores in result["evaluator"]["scores"]:
                        metric = metric_scores["name"]
                        score = metric_scores["score"]
                        data_list.append(
                            {
                                "run": run_name,
                                "model": model_name,
                                "dataset": dataset_name,
                                "metric": metric,
                                "score": score,
                                "task": task_id,
                                "scenario": scenario_id,
                            }
                        )

    return pd.DataFrame(data_list)
