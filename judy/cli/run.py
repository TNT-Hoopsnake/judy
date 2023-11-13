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


def confirm_run():
    click.echo("\nHappy to continue? [y]: ", nl=False)
    c = click.getchar()
    match c:
        case "y":
            click.echo("\n\n")
            return True
        case _:
            click.echo("Cancelling run")
            return False


def summarise_run(num_evaluations, models_to_run, scenarios_to_run, datasets_to_run):
    click.secho("\n\t\t Welcome to Judy! \n\n", bold=True, bg="white", fg="green")

    click.secho("Your pending run will include:\n", bold=True)
    click.secho(f"\t{num_evaluations} evaluations", bold=True)

    click.secho(f"\t{len(models_to_run)} models:", bold=True)
    click.echo("\n".join([f"\t\t{model.id}" for model in models_to_run]))

    click.secho(f"\t{len(scenarios_to_run)} scenarios:", bold=True)
    click.echo("\n".join([f"\t\t{scen.name}" for scen in scenarios_to_run]))

    click.secho(f"\t{len(datasets_to_run)} datasets:", bold=True)
    click.echo("\n".join([f"\t\t{ds}" for ds in datasets_to_run]))


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

    log.info(
        "Running a total of %d evaluations on %d models",
        (len(evaluations_to_run) * run_config.num_evals),
        len(models_to_run),
    )
    num_evaluations = len(evaluations_to_run) * run_config.num_evals

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
