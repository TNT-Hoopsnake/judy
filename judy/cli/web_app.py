import pathlib
import click
from judy.web_app.main import run_webapp


@click.command()
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
def run_web_app(host, port, results_directory):
    if not pathlib.Path(results_directory).is_dir():
        click.echo(f"Invalid directory provided: {results_directory}")
        click.echo("Please ensure it exists and is accessible by Judy")
        return

    run_webapp(host, port, results_directory)
