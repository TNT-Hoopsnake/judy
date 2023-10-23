import pathlib

import click
from dotenv import load_dotenv

from gpt_eval.cache import SqliteCache
from gpt_eval.cli.install import setup_user_dir
from gpt_eval.config import (
    get_config_definitions,
    get_responder_class_map,
    load_and_validate_configs,
)
from gpt_eval.data.loader import load_formatted_data
from gpt_eval.evaluation import Evaluator
from gpt_eval.utils import (
    PromptBuilder,
    get_dataset_config,
    save_evaluation_results,
    ensure_directory_exists,
    dump_configs,
    dump_metadata
)


class EvalCommandLine:
    def __init__(self, force):
        load_dotenv()
        setup_user_dir()
        self.cache = SqliteCache()
        self.force = force

    def get_evaluation_results(
        self, eval_prompts, cache_key, model, eval_config, metrics
    ):
        eval_results_cache_key = (
            f"{self.cache.calculate_content_hash(eval_prompts)}-{model.name}"
        )
        if eval_results := self.cache.get(cache_key, eval_results_cache_key):
            print("Evaluation results retrieved from cache")
        else:
            print("Evaluation results not present in cache")
            evaluator = Evaluator(eval_config=eval_config, metrics=metrics)
            eval_results = evaluator.run_evaluation(eval_prompts)
            self.cache.set(cache_key, eval_results_cache_key, eval_results)

        return eval_results

    def get_formatted_data(self, cache_key, ds_config, eval_config):
        if data := self.cache.get(cache_key, "data"):
            print("Formatted data retrieved from cache")
        else:
            print("Formatted data not present in cache")

            data = load_formatted_data(
                ds_config, eval_config.num_evals, eval_config.random_seed, self.force
            )
            self.cache.set(cache_key, "data", data)

        return data

    def get_evaluation_prompts(
        self, cache_key, model, prompt_builder, ds_config, eval_config, scenario_type
    ):
        eval_prompts_cache_key = f"eval_prompts-{model.name}"

        if eval_prompts := self.cache.get(cache_key, eval_prompts_cache_key):
            print("Evaluation prompts retrieved from cache")

        else:
            print("Evaluation prompts not present in cache")

            data = self.get_formatted_data(cache_key, ds_config, eval_config)

            responder_cls = get_responder_class_map().get(scenario_type)
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
    def get_metrics_for_scenario(scenario, metric_configs):
        metrics = []
        for metric_group in metric_configs:
            for metric_config in metric_group.metrics:
                # override group values with metric specific values
                metric_config.scenarios = (
                    metric_config.scenarios or metric_group.scenarios
                )
                metric_config.min = metric_config.min or metric_group.min
                metric_config.max = metric_config.max or metric_group.max
                if scenario in metric_config.scenarios:
                    metrics.append(metric_config)

        return metrics

    @staticmethod
    def matches_tag(config, tag):
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
@click.option(
    "-s", "--scenario", default=None, help="Only run scenarios matching this tag"
)
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
    "--metric-config",
    help="The path to the metric config file.",
    default=None,
    type=click.Path(),
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    show_default=True,
    default=False,
    help="Force datasets to be re-downloaded",
)
def run_eval(
    scenario,
    model,
    dataset,
    name,
    output,
    dataset_config,
    eval_config,
    metric_config,
    force,
):
    """Run evaluations for models using a judge model."""
    cli = EvalCommandLine(force)
    if eval_config and not pathlib.Path(eval_config).is_file():
        raise FileNotFoundError(f"Eval config file does not exist: {eval_config}")

    if dataset_config and not pathlib.Path(dataset_config).is_file():
        raise FileNotFoundError(f"Eval config file does not exist: {dataset_config}")

    if metric_config and not pathlib.Path(metric_config).is_file():
        raise FileNotFoundError(f"Eval config file does not exist: {metric_config}")

    config_definitions = get_config_definitions(
        eval_config, dataset_config, metric_config
    )
    configs = load_and_validate_configs(config_definitions)

    model_tag = model
    dataset_tag = dataset
    scenario_tag = scenario

    eval_config = configs["eval"]
    dataset_configs = configs["datasets"]
    metric_configs = configs["metrics"]

    output_dir = pathlib.Path(output)
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output}")
    results_dir = output_dir / name

    ensure_directory_exists(results_dir)
    dump_configs(results_dir, configs)
    dump_metadata(results_dir, dataset, scenario, model)

    models, scenarios, datasets = set(), set(), set()

    for eval_model in eval_config.evaluated_models:
        if not EvalCommandLine.matches_tag(eval_model, model_tag):
            click.echo(f"Skipping model {eval_model.name} - does not match tag")
            continue
        # use the model specific values if they exist
        # else use the general eval values
        eval_model.temperature = eval_model.temperature or eval_config.temperature
        eval_model.max_tokens = eval_model.max_tokens or eval_config.max_tokens
        eval_model.context_char_limit = (
            eval_model.context_char_limit or eval_config.context_char_limit
        )

        for eval_scenario in eval_config.scenarios:
            if not EvalCommandLine.matches_tag(eval_scenario, scenario_tag):
                click.echo(
                    f"Skipping scenario {eval_scenario.type} - does not match tag"
                )
                continue
            metrics = cli.get_metrics_for_scenario(eval_scenario.type, metric_configs)
            prompt_builder = PromptBuilder(eval_scenario.type, metrics)

            for dataset_name in eval_scenario.datasets:
                ds_config = get_dataset_config(dataset_name, dataset_configs)
                if not EvalCommandLine.matches_tag(ds_config, dataset_tag):
                    click.echo(
                        f"Skipping dataset {ds_config.name} - does not match tag"
                    )
                    continue
                cache_key = cli.cache.build_cache_key(dataset_name, eval_scenario.type)

                eval_prompts = cli.get_evaluation_prompts(
                    cache_key,
                    eval_model,
                    prompt_builder,
                    ds_config,
                    eval_config,
                    eval_scenario.type,
                )

                evaluation_results = cli.get_evaluation_results(
                    eval_prompts, cache_key, eval_model, eval_config, metrics
                )

                all_results = []
                for prompt, result in zip(eval_prompts, evaluation_results):
                    all_results.append({**prompt, **result})
                save_evaluation_results(
                    eval_model.name, dataset_name, all_results, results_dir
                )
                models.add(eval_model.name)
                scenarios.add(eval_scenario.type)
                datasets.add(dataset_name)
    click.echo(
        f"Successfully evaluated {len(models)} models on {len(scenarios)} scenarios using {len(datasets)} datasets"
    )
