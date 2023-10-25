import pathlib
from typing import List

import click
from dotenv import load_dotenv

from gpt_eval.cache import SqliteCache
from gpt_eval.cli.install import setup_user_dir
from gpt_eval.config import (
    get_config_definitions,
    get_responder_class_map,
    load_and_validate_configs,
    EvaluatedModel,
    DatasetConfig,
    TaskConfig,
    TaskTypes,
    MetricGroupConfig,
    EvaluationConfig,
    MetricConfig,
    RunConfig,
)
from gpt_eval.data.loader import load_formatted_data
from gpt_eval.evaluation import Evaluator
from gpt_eval.utils import PromptBuilder, get_dataset_config, save_evaluation_results


class EvalCommandLine:
    def __init__(self):
        load_dotenv()
        setup_user_dir()
        self.cache = SqliteCache()

    def get_evaluation_results(
        self,
        eval_prompts: List[dict],
        cache_key: str,
        model: str,
        run_config: RunConfig,
        metrics: List[MetricConfig],
        ignore_cache: bool,
    ):
        eval_results = None
        eval_results_cache_key = (
            f"{self.cache.calculate_content_hash(eval_prompts)}-{model.name}"
        )
        if ignore_cache:
            print("Skipping cache")
        else:
            eval_results = self.cache.get(cache_key, eval_results_cache_key)
            if not eval_results:
                print("Evaluation results not present in cache")

        if eval_results:
            print("Evaluation results retrieved from cache")
        else:
            evaluator = Evaluator(run_config=run_config, metrics=metrics)
            eval_results = evaluator.run_evaluation(eval_prompts)
            self.cache.set(cache_key, eval_results_cache_key, eval_results)

        return eval_results

    def get_formatted_data(
        self,
        cache_key: str,
        ds_config: DatasetConfig,
        run_config: RunConfig,
        ignore_cache: bool,
    ):
        data = None
        if ignore_cache:
            print("Skipping cache")
        else:
            data = self.cache.get(cache_key, "data")
            if not data:
                print("Formatted data not present in cache")

        if data:
            print("Formatted data retrieved from cache")
        else:
            data = load_formatted_data(
                ds_config, run_config.num_evals, run_config.random_seed, ignore_cache
            )
            self.cache.set(cache_key, "data", data)

        return data

    def get_evaluation_prompts(
        self,
        cache_key: str,
        model: str,
        prompt_builder: PromptBuilder,
        ds_config: DatasetConfig,
        run_config: RunConfig,
        task_type: TaskTypes,
        ignore_cache: bool,
    ) -> List[dict]:
        eval_prompts_cache_key = f"eval_prompts-{model.name}"
        eval_prompts = None
        if ignore_cache:
            print("Skipping cache")
        else:
            eval_prompts = self.cache.get(cache_key, eval_prompts_cache_key)
            if not eval_prompts:
                print("Evaluation prompts not present in cache")

        if eval_prompts:
            print("Evaluation prompts retrieved from cache")
        else:
            data = self.get_formatted_data(
                cache_key, ds_config, run_config, ignore_cache
            )

            responder_cls = get_responder_class_map().get(task_type)
            # sanity check
            if not responder_cls:
                raise ValueError("Unable to determine responder class")

            responder = responder_cls(
                data=data,
                prompt_builder=prompt_builder,
                model_config=model,
            )
            eval_prompts = responder.get_evaluation_prompts()

            self.cache.set(cache_key, eval_prompts_cache_key, eval_prompts)

        return eval_prompts

    @staticmethod
    def get_tasks_for_run(run_config: RunConfig, eval_config: EvaluationConfig):
        tasks = []

        for task_id in run_config.tasks:
            # Ensure selected task is defined in the eval config
            matching_task = next(
                filter(lambda task: task.id == task_id, eval_config.tasks),
                None,
            )
            assert (
                matching_task,
                f"Task {task_id} is undefined. Create an entry for it in the evaluation config",
            )
            tasks.append(matching_task)
        return tasks

    @staticmethod
    def get_metric_groups_for_run(run_config: RunConfig, eval_config: EvaluationConfig):
        metrics = []
        for metric_id in run_config.metrics:
            matching_metric_group = next(
                filter(lambda metric: metric.id == metric_id, eval_config.metric_groups)
            )
            # Ensure selected task is defined in the eval config
            assert (
                matching_metric_group,
                f"Metric group {metric_id} is undefined. Create an entry for it in the evaluation config",
            )
            metrics.append(matching_metric_group)
        return metrics

    @staticmethod
    def get_metrics_for_group(task_id: TaskTypes, metric_group: str):
        metrics = []
        for metric_config in metric_group.metrics:
            # override group values with metric specific values
            metric_config.tasks = metric_config.tasks or metric_group.tasks
            if task_id not in metric_config.tasks:
                continue
            metric_config.min = metric_config.min or metric_group.min
            metric_config.max = metric_config.max or metric_group.max
            metrics.append(metric_config)
        return metrics

    @staticmethod
    def get_metrics_for_task(
        task_id: TaskTypes,
        run_metric_groups: List[MetricGroupConfig],
        eval_metric_groups: List[MetricGroupConfig],
    ):
        metrics = []
        # 1. Use metrics defined in the run config for the task
        for metric_group in run_metric_groups:
            metrics.extend(EvalCommandLine.get_metrics_for_group(task_id, metric_group))

        # 2. If no metrics match, use all matching metrics for the task from the eval config
        if not metrics:
            for metric_group in eval_metric_groups:
                metrics.extend(
                    EvalCommandLine.get_metrics_for_group(task_id, metric_group)
                )
        return metrics

    @staticmethod
    def matches_tag(
        config: TaskConfig | EvaluatedModel | DatasetConfig, tag: str
    ) -> bool:
        if not tag or not hasattr(config, "tags"):
            return True
        config_tags = config.tags or []
        if tag in config_tags:
            return True
        return False


@click.command()
@click.option(
    "-t", "--dataset", default=None, help="Only run datasets matching this tag"
)
@click.option("-s", "--task", default=None, help="Only run tasks matching this tag")
@click.option("-m", "--model", default=None, help="Only run models matching this tag")
@click.option(
    "-n",
    "--name",
    default="default",
    help="A unique identifier to group the evaluation results. Overrides existing results with the same name.",
)
@click.option(
    "-o",
    "--output",
    help="The path to a directory to save evaluation results.",
    type=click.Path(),
    required=True,
)
@click.option(
    "-d",
    "--dataset-config",
    help="The path to the dataset config file.",
    default=None,
    type=click.Path(),
)
@click.option(
    "-e",
    "--eval-config",
    help="The path to the eval config file.",
    default=None,
    type=click.Path(),
)
@click.option(
    "-r",
    "--run-config",
    help="The path to the run config file.",
    type=click.Path(),
)
@click.option(
    "-f",
    "--ignore-cache",
    is_flag=True,
    show_default=True,
    default=False,
    help="Ignore cache and force datasets to be re-downloaded and prompts to be re-evaluated",
)
def run_eval(
    task,
    model,
    dataset,
    name,
    output,
    dataset_config,
    eval_config,
    run_config,
    ignore_cache,
):
    """Run evaluations for models using a judge model."""
    cli = EvalCommandLine()
    if eval_config and not pathlib.Path(eval_config).is_file():
        raise FileNotFoundError(f"Eval config file does not exist: {eval_config}")

    if dataset_config and not pathlib.Path(dataset_config).is_file():
        raise FileNotFoundError(f"Eval config file does not exist: {dataset_config}")

    if run_config and not pathlib.Path(run_config).is_file():
        raise FileNotFoundError(f"Run config file does not exist: {run_config}")

    config_definitions = get_config_definitions(eval_config, dataset_config, run_config)
    configs = load_and_validate_configs(config_definitions)

    model_tag = model
    dataset_tag = dataset
    task_tag = task

    eval_config = configs["eval"]
    dataset_config = configs["datasets"]
    run_config = configs["run"]

    output_dir = pathlib.Path(output)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output}")
    results_dir = output_dir / name

    models, tasks, datasets = set(), set(), set()

    run_tasks = cli.get_tasks_for_run(run_config, eval_config)
    run_metric_groups = cli.get_metric_groups_for_run(run_config, eval_config)

    for eval_model in eval_config.models:
        if not EvalCommandLine.matches_tag(eval_model, model_tag):
            click.echo(f"Skipping model {eval_model.id} - does not match tag")
            continue
        # use the model specific values if they exist
        # else use the general eval values
        eval_model.temperature = eval_model.temperature or run_config.temperature
        eval_model.max_tokens = eval_model.max_tokens or run_config.max_tokens
        eval_model.context_char_limit = (
            eval_model.context_char_limit or run_config.context_char_limit
        )

        for eval_task in run_tasks:
            if not EvalCommandLine.matches_tag(eval_task, task_tag):
                click.echo(f"Skipping task {eval_task.id} - does not match tag")
                continue
            metrics = cli.get_metrics_for_task(
                eval_task.id, run_metric_groups, eval_config.metric_groups
            )
            prompt_builder = PromptBuilder(eval_task.id, metrics)

            for dataset_id in eval_task.datasets:
                ds_config = get_dataset_config(dataset_id, dataset_config)
                if not EvalCommandLine.matches_tag(ds_config, dataset_tag):
                    click.echo(f"Skipping dataset {ds_config.id} - does not match tag")
                    continue

                cache_key = cli.cache.build_cache_key(dataset_id, eval_task.id)

                eval_prompts = cli.get_evaluation_prompts(
                    cache_key,
                    eval_model,
                    prompt_builder,
                    ds_config,
                    run_config,
                    eval_task.id,
                    ignore_cache,
                )

                evaluation_results = cli.get_evaluation_results(
                    eval_prompts,
                    cache_key,
                    eval_model,
                    run_config,
                    metrics,
                    ignore_cache,
                )

                save_evaluation_results(
                    eval_model.id,
                    dataset_id,
                    eval_prompts,
                    evaluation_results,
                    results_dir,
                )
                models.add(eval_model.id)
                tasks.add(eval_task.id)
                datasets.add(dataset_id)
    click.echo(
        f"Successfully evaluated {len(models)} models on {len(tasks)} tasks using {len(datasets)} datasets"
    )
