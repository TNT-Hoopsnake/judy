import os
import json

from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from judy.config import (
    EvaluationConfig,
    DatasetConfig,
    RunConfig,
    load_validated_config,
)
from judy.utils import matches_tag

CONFIG_CLASS_MAP = {
    "eval": EvaluationConfig,
    "datasets": DatasetConfig,
    "run": RunConfig,
}
from collections.abc import MutableMapping as Map


def nested_update(d, v):
    """
    Nested update of dict-like 'd' with dict-like 'v'.
    """

    for key in v:
        if key in d and isinstance(d[key], Map) and isinstance(v[key], Map):
            nested_update(d[key], v[key])
        else:
            d[key] = v[key]

    return d


def check_directory_contains_subdirectories(directory):
    directory_path = Path(directory)
    return any(sub_path.is_dir() for sub_path in directory_path.iterdir())


def load_json(path):
    with open(path, "r") as fn:
        try:
            return json.load(fn)
        except ValueError:
            print(f"Error loading json data from path {path}")
            return None


def load_configs(config_path) -> dict:
    """
    Load configuration data from a JSON file and validate it.
    This loads (Run, Eval, Dataset) configurations into a config dict

    Parameters:
        config_path (str): The path to the configuration JSON file.

    Returns:
        dict or None: A dictionary containing validated configuration data, or None if an error occurs.
    """
    configs = {}
    data = load_json(config_path)

    for key, config in data.items():
        config_cls = CONFIG_CLASS_MAP.get(key)
        if not config_cls:
            print(f"Error retrieving config class using key {key}")
            return None

        config_model = load_validated_config(config, config_cls)
        configs[key] = config_model

    return configs


def get_run_data_index(configs, metadata):
    """
    Create an index of datasets, tasks, scenarios, and models which we used in a specific run

    Parameters:
        configs (dict): Configuration data for runs.
        metadata (dict): Metadata associated with the run.

    Returns:
        dict: An index containing datasets, tasks, scenarios, and models.
    """
    index = {"datasets": {}, "tasks": {}, "scenarios": {}, "models": {}}

    for dataset in configs["datasets"]:
        if matches_tag(dataset, metadata["dataset_tags"]):
            index["datasets"][dataset.id] = dataset

    for task in configs["eval"].tasks:
        if matches_tag(task, metadata["task_tags"]):
            index["tasks"][task.id] = task

    for scenario in configs["eval"].scenarios:
        if scenario.id in configs["run"].scenarios:
            index["scenarios"][scenario.id] = scenario

    for model in configs["run"].models:
        if matches_tag(model, metadata["model_tags"]):
            index["models"][model.id] = model

    return index


def check_directory(directory: str | Path):
    directory = Path(directory)
    if not directory.exists():
        return False
    if not directory.is_dir():
        return False
    return True


def get_all_data_index(data_directory):
    """
    Create an index of datasets, tasks, scenarios, and models which we used across all runs run

    Parameters:
        data_directory (str): The path to the directory containing run data.

    Returns:
        dict: An index containing datasets, tasks, scenarios, and models.
    """
    index = {"datasets": {}, "tasks": {}, "scenarios": {}, "models": {}}

    for run_name in os.listdir(data_directory):
        run_path = os.path.join(data_directory, run_name)
        if not check_directory_contains_subdirectories(run_path):
            # this run directory has no subdirectories
            # ie: no results exist for this run
            print(f"Warning: Empty results directory located at {run_path}.")
            continue
        config = load_configs(os.path.join(run_path, "config.json"))
        if not config:
            print(
                f"Configurations could not be loaded for run ({run_name}). It will be skipped."
            )
            continue

        for task in config["eval"].tasks:
            index["tasks"][task.id] = task

        for dataset in config["datasets"]:
            index["datasets"][dataset.id] = dataset

        for scenario in config["eval"].scenarios:
            index["scenarios"][scenario.id] = scenario

        for model in config["run"].models:
            index["models"][model.id] = model

    return index


def load_all_data(data_directory):
    all_models_used = set()
    all_tasks_used = set()
    all_scenarios_used = set()
    all_datasets_used = set()

    runs_data = {}
    runs_df = []
    total_evaluations = 0
    all_data_indexes = {}
    # Iterate over the run names in a directory
    for run_directory in Path(data_directory).iterdir():
        run_name = run_directory.stem

        if not check_directory_contains_subdirectories(run_directory):
            # this run directory has no model results
            # nothing to do here - skip!
            continue

        run_results_data = {}
        models_used = set()
        tasks_used = set()
        scenarios_used = set()
        datasets_used = set()
        run_config = load_configs(run_directory / "config.json")
        run_metadata = load_json(run_directory / "metadata.json")

        run_data_index = {}
        total_run_evaluations = 0
        for model_directory in run_directory.iterdir():
            if not check_directory(model_directory):
                continue

            model_config = load_configs(model_directory / "config.json")
            if not model_config:
                # cant do anything without the loaded config
                continue
            model_metadata = load_json(model_directory / "metadata.json")

            model_dataset_directory = model_directory / "datasets"
            if not check_directory(model_dataset_directory):
                continue

            model_id = model_metadata["model_id"]
            models_used.add(model_id)

            model_results_data = {}
            for results_file in model_dataset_directory.iterdir():
                if not results_file.is_file():
                    # this should be a file. cant do anything if its not
                    print(f"not a file - {results_file}")
                    continue

                results_data = load_json(results_file)
                for result in results_data:
                    scenario_id = result["scenario_id"]
                    dataset_id = result["dataset_id"]
                    task_id = result["task_id"]

                    scenario_results_data = model_results_data.get(scenario_id, {})
                    dataset_results_data = scenario_results_data.get(dataset_id, {})
                    task_results_data = dataset_results_data.get(task_id, [])
                    task_results_data.append(result)

                    dataset_results_data[task_id] = task_results_data
                    scenario_results_data[dataset_id] = dataset_results_data
                    model_results_data[scenario_id] = scenario_results_data

                    scenarios_used.add(scenario_id)
                    datasets_used.add(dataset_id)
                    tasks_used.add(task_id)

                    total_evaluations += 1
                    total_run_evaluations += 1

                    for metric_scores in result["evaluator"]["scores"]:
                        metric = metric_scores["name"]
                        score = metric_scores["score"]
                        runs_df.append(
                            {
                                "run": run_name,
                                "model": model_id,
                                "dataset": dataset_id,
                                "metric": metric,
                                "score": score,
                                "task": task_id,
                                "scenario": scenario_id,
                            }
                        )

            model_data = {
                "config": model_config,
                "metadata": model_metadata,
                "data": model_results_data,
            }
            model_data_index = get_run_data_index(model_config, model_metadata)
            run_data_index = nested_update(run_data_index, model_data_index)
            run_results_data[model_id] = model_data

        runs_data[run_name] = {
            "results": run_results_data,
            "models_used": list(models_used),
            "tasks_used": list(tasks_used),
            "datasets_used": list(datasets_used),
            "scenarios_used": list(scenarios_used),
            "metadata": run_metadata,
            "config": run_config,
            "data_index": run_data_index,
            "total_evaluations": total_run_evaluations,
        }
        all_data_indexes = nested_update(all_data_indexes, run_data_index)
        all_models_used = all_models_used.union(models_used)
        all_tasks_used = all_tasks_used.union(tasks_used)
        all_datasets_used = all_datasets_used.union(datasets_used)
        all_scenarios_used = all_scenarios_used.union(scenarios_used)

    all_runs_data = {
        "runs": runs_data,
        "models_used": list(all_models_used),
        "tasks_used": list(all_tasks_used),
        "datasets_used": list(all_datasets_used),
        "scenarios_used": list(all_scenarios_used),
        "total_evaluations": total_evaluations,
        "data_index": all_data_indexes,
    }

    return all_runs_data, pd.DataFrame(runs_df)


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


def get_heatmap_data(df, group_by_field):
    grouped = df.groupby(group_by_field)
    grouped_data = {}

    for model, data in grouped:
        pivoted_data = pd.pivot_table(
            data, index="model", columns="metric", values="score", aggfunc="mean"
        )
        pivoted_data = pivoted_data.round(2).replace(np.nan, "-")

        model_names = pivoted_data.index.to_list()
        metric_names = pivoted_data.columns.to_list()

        heatmap_data = []
        for model_name in model_names:
            for metric in metric_names:
                val = pivoted_data.loc[model_name, metric]
                # Convert numpy types to python types to enable json serialization
                # Metrics can only be number types
                if hasattr(val, "dtype") and val.dtype.name.startswith("int"):
                    val = int(val)
                elif hasattr(val, "dtype") and val.dtype.name.startswith("float"):
                    val = float(val)
                heatmap_data.append([metric, model_name, val])

        grouped_data[model] = {
            "model_names": model_names,
            "metric_names": metric_names,
            "heatmap_data": heatmap_data,
        }

    return grouped_data
