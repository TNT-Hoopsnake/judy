import json
import os
from typing import List
from datetime import datetime
from gpt_eval.config import (
    DatasetConfig,
    TaskConfig,
    EvaluatedModel,
)
from gpt_eval.responders import EvalPrompt, EvalResponse



def matches_tag(config: TaskConfig | EvaluatedModel | DatasetConfig, tag: str) -> bool:
    if not tag or not hasattr(config, "tags"):
        return True
    config_tags = config.tags or []
    if tag in config_tags:
        return True
    return False


def ensure_directory_exists(dir_path: str) -> str:
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path


def save_evaluation_results(
    model_name: str,
    scenario_id: str,
    task_id: str,
    dataset_name: str,
    eval_prompts: List[EvalPrompt],
    eval_results: List[EvalResponse],
    results_dir: str,
):
    ensure_directory_exists(results_dir)

    model_results_dir = ensure_directory_exists(os.path.join(results_dir, model_name))
    clean_ds_name = dataset_name.split("/")[-1]
    data = []
    for idx, item in enumerate(eval_results):
        model = {
            "response": eval_prompts[idx].model_response.response,
            **eval_prompts[idx].model_response.prompt.model_dump(),
        }
        data.append(
            {
                "task_id": task_id,
                "scenario_id":scenario_id,
                "model": model,
                "evaluator": item.model_dump(mode="json"),
            }
        )

    with open(
        os.path.join(model_results_dir, f"{clean_ds_name}-results.json"), "w+"
    ) as fn:
        json.dump(data, fn, indent=4)


def dump_metadata(
    dir_path: str, dataset_tags: List[str], task_tags: List[str], model_tags: List[str]
):
    with open(dir_path / "metadata.json", "w+") as fn:
        data = {
            "dataset_tags": dataset_tags,
            "task_tags": task_tags,
            "model_tags": model_tags,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        json.dump(data, fn, indent=4)
