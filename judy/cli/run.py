import pathlib
import click
from tqdm import tqdm

from judy.cli.manager import EvalManager
from judy.config import (
    dump_configs,
    DATASET_CONFIG_PATH,
    EVAL_CONFIG_PATH,
    RUN_CONFIG_PATH,
)
from judy.utils import (
    PromptBuilder,
    save_evaluation_results,
    dump_metadata,
    load_configs,
    get_output_directory,
)
from judy.config.logging import logger as log
from judy.web_app.main import run_webapp


@click.group()
def judy_cli():
    click.secho("\n\t\t Welcome to Judy! \n\n", bold=True)


def confirm_run():
    click.echo("\nHappy to continue? [y]: ", nl=False)
    c = click.getchar().lower()
    match c:
        case "y":
            click.echo("\n\n")
            return True
        case _:
            click.echo("Cancelling run")
            return False


def summarise_run(num_evaluations, models_to_run, scenarios_to_run, datasets_to_run):
    click.secho("Your pending run will include:\n", bold=True)
    click.secho(f"\t{num_evaluations} evaluations", bold=True)

    click.secho(f"\t{len(models_to_run)} models:", bold=True)
    click.echo("\n".join([f"\t\t{model.id}" for model in models_to_run]))

    click.secho(f"\t{len(scenarios_to_run)} scenarios:", bold=True)
    click.echo("\n".join([f"\t\t{scen.name}" for scen in scenarios_to_run]))

    click.secho(f"\t{len(datasets_to_run)} datasets:", bold=True)
    click.echo("\n".join([f"\t\t{ds}" for ds in datasets_to_run]))


@judy_cli.command()
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
    default=DATASET_CONFIG_PATH,
    type=click.Path(),
)
@click.option(
    "-e",
    "--eval-config-path",
    help="The path to the eval config file.",
    default=EVAL_CONFIG_PATH,
    type=click.Path(),
)
@click.option(
    "-r",
    "--run-config-path",
    help="The path to the run config file.",
    default=RUN_CONFIG_PATH,
    type=click.Path(),
)
@click.option(
    "-f",
    "--ignore-cache",
    type=click.Choice(["all", "datasets", "prompts"], case_sensitive=False),
    default=None,
    help="Ignore cache and force datasets to be re-downloaded and/or prompts to be re-evaluated",
)
@click.option(
    "-c",
    "--clear-cache",
    is_flag=True,
    default=False,
    help="Destroy all data contained within the cache DB",
)
def run(
    task,
    model,
    dataset,
    name,
    output,
    dataset_config_path,
    eval_config_path,
    run_config_path,
    ignore_cache,
    clear_cache,
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

    manager = EvalManager([eval_config_path, run_config_path], clear_cache)
    # Collect evaluations to run
    evaluations_to_run, scenarios_to_run, config_cache = manager.collect_evaluations(
        run_config, eval_config, dataset_config
    )
    models_to_run = manager.get_models_to_run(run_config, model_tag)
    datasets_to_run = list(set({eval[1] for eval in evaluations_to_run}))
    num_evaluations = len(evaluations_to_run) * run_config.num_evals

    log.info(
        "Running a total of %d evaluations on %d models",
        num_evaluations,
        len(models_to_run),
    )

    summarise_run(num_evaluations, models_to_run, scenarios_to_run, datasets_to_run)

    num_cache_items = manager.sizeof_current_run_cache()
    if not ignore_cache:
        if num_cache_items > 0:
            click.echo(f"\nThis run will use {num_cache_items} values from the cache")
        else:
            click.echo("\nNo values have been cached for this run")
    else:
        click.echo(
            f"\n{str(ignore_cache).rstrip('s').title()} cache values will be ignored"
        )

    if not confirm_run():
        return

    with tqdm(total=len(models_to_run) * len(evaluations_to_run)) as pbar:
        for eval_model in models_to_run:
            log.info("Evaluation started for model: %s", eval_model.id)
            # Model-specific parameters override general run config parameters
            eval_model.temperature = eval_model.temperature or run_config.temperature
            eval_model.max_tokens = eval_model.max_tokens or run_config.max_tokens
            eval_model.context_char_limit = (
                eval_model.context_char_limit or run_config.context_char_limit
            )
            for scenario_id, dataset_id, task_id in evaluations_to_run:
                pbar.set_description(f"Processing: {eval_model.id} - {dataset_id}")
                ds_config = config_cache["datasets"].get(dataset_id)
                scenario_metrics = config_cache["scenario_metrics"].get(scenario_id)
                prompt_builder = PromptBuilder(task_id, scenario_metrics)

                cache_key = manager.cache.build_cache_key(dataset_id, task_id)

                log.info(
                    "Running scenario: %s, dataset: %s, task: %s",
                    scenario_id,
                    dataset_id,
                    task_id,
                )

                eval_prompts = manager.get_evaluation_prompts(
                    cache_key,
                    eval_model,
                    prompt_builder,
                    ds_config,
                    run_config,
                    task_id,
                    ignore_cache,
                )

                evaluation_results = manager.get_evaluation_results(
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
                # advance progress bar when an eval has been completed
                pbar.update(1)

    log.info(
        "Successfully ran %d evaluations on %d models",
        (len(evaluations_to_run) * run_config.num_evals),
        len(models_to_run),
    )


@judy_cli.command()
@click.option(
    "-h", "--host", default="127.0.0.1", type=str, help="Host used for the web app"
)
@click.option("-p", "--port", default=5000, type=int, help="Port used for the web app")
@click.option(
    "-r",
    "--results-directory",
    help="The path to the directory that contains results that should be loaded by the web app",
    type=click.Path(),
    required=True,
)
def serve(host, port, results_directory):
    """Run a flask web app for displaying results"""
    if not pathlib.Path(results_directory).is_dir():
        click.echo(f"Invalid directory provided: {results_directory}")
        click.echo("Please ensure it exists and is accessible by Judy")
        return

    run_webapp(host, port, results_directory)


@judy_cli.command()
@click.option(
    "-d",
    "--dataset-config-path",
    help="The path to the dataset config file.",
    default=DATASET_CONFIG_PATH,
    type=str,
)
@click.option(
    "-e",
    "--eval-config-path",
    help="The path to the eval config file.",
    default=EVAL_CONFIG_PATH,
    type=str,
)
@click.option(
    "-r",
    "--run-config-path",
    help="The path to the run config file.",
    default=RUN_CONFIG_PATH,
    type=str,
)
def config(dataset_config_path, eval_config_path, run_config_path):
    """View/Edit configuration files"""

    for config_path in [dataset_config_path, eval_config_path, run_config_path]:
        if not pathlib.Path(config_path).is_file():
            click.echo(f"Invalid path provided. No file found at {config_path}")
            return

    quit_requested = False
    while not quit_requested:
        click.echo("\nYour config files are defined in the following locations:\n")
        click.echo(f"\t(R)un Config: {run_config_path}")
        click.echo(f"\t(E)valuation Config: {eval_config_path}")
        click.echo(f"\t(D)ataset Config: {dataset_config_path}")
        click.echo(
            "\nPlease input R, E or D (case-insensitive) to choose which config file you would like to open.\nPress any other key to quit: ",
            nl=False,
        )
        c = click.getchar().lower()
        match c:
            case "r":
                click.echo("\nOpening Run config file")
                click.edit(filename=run_config_path)
            case "e":
                click.echo("\nOpening Evaluation config file")
                click.edit(filename=eval_config_path)
            case "d":
                click.echo("\nOpening Dataset config file")
                click.edit(filename=dataset_config_path)
            case _:
                click.echo("\nQuitting...")
                quit_requested = True

        click.clear()
