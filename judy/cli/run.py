import asyncio
import pathlib
import click

from judy.cli.manager import EvalManager
from judy.cli.install import setup_user_dir
from judy.config import (
    dump_configs,
    DATASET_CONFIG_PATH,
    EVAL_CONFIG_PATH,
    RUN_CONFIG_PATH,
)
from judy.utils import (
    dump_metadata,
    load_configs,
    get_output_directory,
)
from judy.config.logging import logger as log
from judy.web_app.main import run_webapp


@click.group()
def judy_cli():
    click.secho("\n\t\t Welcome to Judy! \n\n", bold=True)
    setup_user_dir()


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


async def run_evaluations(
    manager: EvalManager,
    yes: bool = False,
):
    # Collect evaluations to run
    evaluations_to_run, scenarios_to_run = manager.collect_evaluations()
    models_to_run = manager.get_models_to_run(manager.run_config, manager.model_tag)
    datasets_to_run = list(set({eval[1] for eval in evaluations_to_run}))
    num_evaluations = len(evaluations_to_run) * manager.run_config.num_evals

    log.info(
        "Running a total of %d evaluations on %d models",
        num_evaluations,
        len(models_to_run),
    )

    summarise_run(num_evaluations, models_to_run, scenarios_to_run, datasets_to_run)

    num_cache_items = manager.sizeof_current_run_cache()
    if not manager.ignore_cache:
        if num_cache_items > 0:
            click.echo(f"\nThis run will use {num_cache_items} values from the cache")
        else:
            click.echo("\nNo values have been cached for this run")
    else:
        click.echo(
            f"\n{str(manager.ignore_cache).rstrip('s').title()} cache values will be ignored"
        )

    # Handle automatic and manual user confirmation
    if not yes and not confirm_run():
        return

    # always dump the latest version of the run config and metadata
    dump_configs(manager.results_dir, manager.configs)
    dump_metadata(
        manager.results_dir, manager.dataset_tag, manager.task_tag, manager.model_tag
    )
    num_evals = len(models_to_run) * num_evaluations
    manager.initialize_progress_bars(num_evals)

    # Run the async evaluation pipeline
    await manager.run_pipeline(models_to_run, evaluations_to_run)

    # Print any exceptions that were raised
    if manager.exceptions:
        log.error("The following exceptions were raised during evaluation:")
        import traceback

        for exc in manager.exceptions:
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    log.info(
        "Successfully ran %d evaluations on %d models",
        (len(evaluations_to_run) * manager.run_config.num_evals),
        len(models_to_run),
    )


@judy_cli.command()
@click.option(
    "-dt", "--dataset", default=None, help="Only run datasets matching this tag"
)
@click.option("-tt", "--task", default=None, help="Only run tasks matching this tag")
@click.option("-mt", "--model", default=None, help="Only run models matching this tag")
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
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    default=False,
    help="Run all evaluations without confirmation",
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
    yes,
):
    """Run evaluations for models using a judge model."""

    configs = load_configs(eval_config_path, dataset_config_path, run_config_path)

    # Create output directory
    tags = {"model": model, "dataset": dataset, "task": task}
    results_dir = get_output_directory(output, name)
    manager = EvalManager(
        tags,
        configs,
        results_dir,
        [eval_config_path, run_config_path],
        clear_cache,
        ignore_cache,
    )

    asyncio.run(
        run_evaluations(manager, yes),
        debug=True,
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
