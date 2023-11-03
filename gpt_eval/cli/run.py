import pathlib
from typing import List, Optional

import click
from dotenv import load_dotenv

from gpt_eval.cache import SqliteCache
from gpt_eval.cli.install import setup_user_dir
from gpt_eval.config import (
    get_config_definitions,
    get_dataset_config,
    get_task_config,
    dump_configs,
    load_and_validate_configs,
    EvaluatedModel,
    DatasetConfig,
    TaskConfig,
    TaskTypes,
    ScenarioConfig,
    EvaluationConfig,
    MetricConfig,
    RunConfig,
    IgnoreCacheTypes,
    EVAL_CONFIG_PATH,
    RUN_CONFIG_PATH,
    DATASET_CONFIG_PATH,
)
from gpt_eval.dataset import BaseFormattedData
from gpt_eval.dataset.loader import load_formatted_data
from gpt_eval.evaluation import Evaluator
from gpt_eval.responders import get_responder_class_map
from gpt_eval.utils import (
    PromptBuilder,
    save_evaluation_results,
    dump_metadata,
    ensure_directory_exists,
)


class EvalCommandLine:
    def __init__(self, config_paths: List[str | pathlib.Path] = None):
        load_dotenv()
        setup_user_dir()
        self.cache = SqliteCache(config_paths)

    def get_evaluation_results(
        self,
        eval_prompts: List[dict],
        cache_key: str,
        model: str,
        run_config: RunConfig,
        metrics: List[MetricConfig],
        ignore_cache_type: IgnoreCacheTypes,
    ):
        eval_results = None
        eval_results_cache_key = (
            f"{self.cache.calculate_content_hash(eval_prompts)}-{model.name}"
        )
        if ignore_cache_type and ignore_cache_type in [
            IgnoreCacheTypes.ALL,
            IgnoreCacheTypes.PROMPTS,
        ]:
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
    ) -> BaseFormattedData:
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
        ignore_cache_type: IgnoreCacheTypes,
    ) -> List[dict]:
        eval_prompts_cache_key = f"eval_prompts-{model.name}"
        eval_prompts = None
        if ignore_cache_type and ignore_cache_type in [
            IgnoreCacheTypes.ALL,
            IgnoreCacheTypes.PROMPTS,
        ]:
            print("Skipping cache")
        else:
            eval_prompts = self.cache.get(cache_key, eval_prompts_cache_key)
            if not eval_prompts:
                print("Evaluation prompts not present in cache")

        if eval_prompts:
            print("Evaluation prompts retrieved from cache")
        else:
            ignore_dataset_cache: bool = ignore_cache_type in [
                IgnoreCacheTypes.ALL,
                IgnoreCacheTypes.DATASET,
            ]
            data = self.get_formatted_data(
                cache_key, ds_config, run_config, ignore_dataset_cache
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
    def collect_evaluations(
        run_config: RunConfig,
        eval_config: EvaluationConfig,
        dataset_config_list: List[DatasetConfig],
    ):
        evaluations_to_run = []
        config_cache = {
            "scenario_metrics": {},
            "datasets": {},
        }
        scenarios_to_run: List[ScenarioConfig] = EvalCommandLine.get_scenarios_for_run(
            run_config, eval_config
        )
        for eval_scenario in scenarios_to_run:
            metrics = EvalCommandLine.get_metrics_for_scenario(eval_scenario)
            config_cache["scenario_metrics"].setdefault(eval_scenario.id, metrics)
            for dataset_id in eval_scenario.datasets:
                dataset = get_dataset_config(dataset_id, dataset_config_list)
                for task_id in dataset.tasks:
                    task = get_task_config(task_id, eval_config)
                    evaluations_to_run.append((eval_scenario.id, dataset.id, task.id))
                    config_cache["datasets"].setdefault(dataset.id, dataset)
        return evaluations_to_run, config_cache

    @staticmethod
    def get_scenarios_for_run(
        run_config: RunConfig,
        eval_config: EvaluationConfig,
        task_tag: Optional[str] = None,
    ) -> List[ScenarioConfig]:
        scenarios = []
        for scenario_id in run_config.scenarios:
            matching_scenario = next(
                filter(
                    lambda scenario: scenario.id
                    == scenario_id,  # pylint: disable=cell-var-from-loop
                    eval_config.scenarios,
                )
            )
            # Ensure selected task is defined in the eval config
            assert (
                matching_scenario
            ), f"Scenario {scenario_id} is undefined. Create an entry for it in the evaluation config"
            if not EvalCommandLine.matches_tag(matching_scenario, task_tag):
                click.echo(f"Skipping task {matching_scenario.id} - does not match tag")
                continue
            scenarios.append(matching_scenario)
        return scenarios

    @staticmethod
    def get_metrics_for_scenario(scenario: ScenarioConfig):
        metrics = []
        for metric_config in scenario.metrics:
            # override group values with metric specific values
            metric_config.score_min = metric_config.score_min or scenario.score_min
            metric_config.score_max = metric_config.score_max or scenario.score_max
            metrics.append(metric_config)
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
    "--dataset-config-path",
    help="The path to the dataset config file.",
    default=None,
    type=click.Path(),
)
@click.option(
    "-e",
    "--eval-config-path",
    help="The path to the eval config file.",
    default=None,
    type=click.Path(),
)
@click.option(
    "-r",
    "--run-config-path",
    help="The path to the run config file.",
    type=click.Path(),
)
@click.option(
    "-f",
    "--ignore-cache",
    type=click.Choice(["all", "datasets", "prompts"], case_sensitive=False),
    default=None,
    help="Ignore cache and force datasets to be re-downloaded and/or prompts to be re-evaluated",
)
def run_eval(
    task,
    model,
    dataset,
    name,
    output,
    dataset_config_path,
    eval_config_path,
    run_config_path,
    ignore_cache,
):
    """Run evaluations for models using a judge model."""
    if eval_config_path and not pathlib.Path(eval_config_path).is_file():
        raise FileNotFoundError(f"Eval config file does not exist: {eval_config_path}")

    if dataset_config_path and not pathlib.Path(dataset_config_path).is_file():
        raise FileNotFoundError(
            f"Eval config file does not exist: {dataset_config_path}"
        )

    if run_config_path and not pathlib.Path(run_config_path).is_file():
        raise FileNotFoundError(f"Run config file does not exist: {run_config_path}")

    eval_config_path = eval_config_path or EVAL_CONFIG_PATH
    dataset_config_path = dataset_config_path or DATASET_CONFIG_PATH
    run_config_path = run_config_path or RUN_CONFIG_PATH

    # Check output directory
    output_dir = pathlib.Path(output)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output}")
    results_dir = output_dir / name
    ensure_directory_exists(results_dir)

    # Validate configs
    config_definitions = get_config_definitions(
        eval_config_path, dataset_config_path, run_config_path
    )
    configs = load_and_validate_configs(config_definitions)

    model_tag = model
    dataset_tag = dataset
    task_tag = task

    eval_config = configs["eval"]
    dataset_config = configs["datasets"]
    run_config = configs["run"]

    dump_configs(results_dir, configs)
    dump_metadata(results_dir, dataset_tag, task_tag, model_tag)

    cli = EvalCommandLine([eval_config_path, run_config_path])
    # Collect evaluations to run
    evaluations_to_run, config_cache = cli.collect_evaluations(
        run_config, eval_config, dataset_config
    )

    models_to_run: List[EvaluatedModel] = [
        model
        for model in run_config.models
        if EvalCommandLine.matches_tag(model, model_tag)
    ]
    click.echo(
        f"Running a total of {len(evaluations_to_run) * run_config.num_evals} evaluations on {len(models_to_run)} models"
    )

    for eval_model in models_to_run:
        click.echo(f"Evaluating {eval_model.id}")
        # Model-specific parameters override general run config parameters
        eval_model.temperature = eval_model.temperature or run_config.temperature
        eval_model.max_tokens = eval_model.max_tokens or run_config.max_tokens
        eval_model.context_char_limit = (
            eval_model.context_char_limit or run_config.context_char_limit
        )
        for scenario_id, dataset_id, task_id in evaluations_to_run:
            ds_config = config_cache["datasets"].get(dataset_id)
            scenario_metrics = config_cache["scenario_metrics"].get(scenario_id)
            prompt_builder = PromptBuilder(task_id, scenario_metrics)

            cache_key = cli.cache.build_cache_key(dataset_id, task_id)

            click.echo(
                f"Running scenario: {scenario_id}, dataset:{dataset_id}, task: {task_id}"
            )

            eval_prompts = cli.get_evaluation_prompts(
                cache_key,
                eval_model,
                prompt_builder,
                ds_config,
                run_config,
                task_id,
                ignore_cache,
            )

            evaluation_results = cli.get_evaluation_results(
                eval_prompts,
                cache_key,
                eval_model,
                run_config,
                scenario_metrics,
                ignore_cache,
            )

            save_evaluation_results(
                eval_model.id,
                dataset_id,
                eval_prompts,
                evaluation_results,
                results_dir,
            )
    click.echo(
        f"Successfully ran {len(evaluations_to_run) * run_config.num_evals} evaluations on {len(models_to_run)} models"
    )
