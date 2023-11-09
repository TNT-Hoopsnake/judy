import logging
import click

from judy.cli.eval_cl import EvalCommandLine
from judy.config import dump_configs
from judy.utils import (
    PromptBuilder,
    save_evaluation_results,
    dump_metadata,
    load_configs,
    get_output_directory,
)


def _init_logger():
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler("app.log")
    formatter = logging.Formatter(
        "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


_init_logger()
_logger = logging.getLogger("app")


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
    configs = load_configs(eval_config_path, dataset_config_path, run_config_path)

    eval_config = configs["eval"]
    dataset_config = configs["datasets"]
    run_config = configs["run"]

    model_tag = model
    dataset_tag = dataset
    task_tag = task

    results_dir = get_output_directory(output, name)

    dump_configs(results_dir, configs)
    dump_metadata(results_dir, dataset_tag, task_tag, model_tag)

    cli = EvalCommandLine([eval_config_path, run_config_path])
    # Collect evaluations to run
    evaluations_to_run, config_cache = cli.collect_evaluations(
        run_config, eval_config, dataset_config
    )
    models_to_run = cli.get_models_to_run(run_config, model_tag)

    _logger.info(
        "Running a total of %d evaluations on %d models",
        (len(evaluations_to_run) * run_config.num_evals),
        len(models_to_run),
    )

    for eval_model in models_to_run:
        _logger.info("Evaluation started for model: %s", eval_model.id)
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

            _logger.info(
                "Running scenario: %s, dataset: %s, task: %s",
                scenario_id,
                dataset_id,
                task_id,
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
                scenario_id,
                task_id,
                dataset_id,
                eval_prompts,
                evaluation_results,
                results_dir,
            )

    _logger.info(
        "Successfully ran %d evaluations on %d models",
        (len(evaluations_to_run) * run_config.num_evals),
        len(models_to_run),
    )
