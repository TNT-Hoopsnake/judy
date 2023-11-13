import pathlib
import click
from judy.config import EVAL_CONFIG_PATH, RUN_CONFIG_PATH, DATASET_CONFIG_PATH


@click.command()
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
def run_edit_configs(dataset_config_path, eval_config_path, run_config_path):
    for config_path in [dataset_config_path, eval_config_path, run_config_path]:
        if not pathlib.Path(config_path).is_file():
            click.echo(f"Invalid path provided. No file found at {config_path}")
            return

    quit_requested = False
    while not quit_requested:
        click.clear()
        click.secho("\n\t\t Welcome to Judy! \n\n", bold=True)

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
