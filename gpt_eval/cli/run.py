import click
from dotenv import load_dotenv
from gpt_eval.evaluation import Evaluator
from gpt_eval.data.loader import load_formatted_data
from gpt_eval.config import (
    get_responder_class_map,
    load_and_validate_configs,
    get_config_definitions
)

from gpt_eval.cache import SqliteCache
from gpt_eval.cli.install import setup_user_dir
from gpt_eval.utils import save_evaluation_results, get_dataset_config, get_dataset_config, PromptBuilder

class EvalCommandLine:

    def __init__(self):
        load_dotenv()
        setup_user_dir()
        self.cache = SqliteCache()

    def get_evaluation_results(self, eval_prompts, cache_key, model, system_config, metrics):
        eval_results_cache_key = f"{self.cache.calculate_content_hash(eval_prompts)}-{model.name}"
        if eval_results := self.cache.get(cache_key, eval_results_cache_key):
            print('Evaluation results retrieved from cache')
        else:
            print('Evaluation results not present in cache')
            evaluator = Evaluator(
                system_config=system_config,
                metrics=metrics
            )
            eval_results = evaluator.run_evaluation(eval_prompts)
            self.cache.set(cache_key, eval_results_cache_key, eval_results)

        return eval_results


    def get_formatted_data(self, cache_key, ds_config, eval_config):
        if data := self.cache.get(cache_key, 'data'):
            print('Formatted data retrieved from cache')
        else:
            print('Formatted data not present in cache')

            data = load_formatted_data(ds_config, eval_config.num_evals, eval_config.random_seed)
            self.cache.set(cache_key, 'data', data)

        return data

    def get_evaluation_prompts(self, cache_key, model, prompt_builder, ds_config, eval_config, scenario_type):
        eval_prompts_cache_key = f'eval_prompts-{model.name}'

        if eval_prompts := self.cache.get(cache_key, eval_prompts_cache_key):
            print('Evaluation prompts retrieved from cache')
        
        else:
            print('Evaluation prompts not present in cache')
            
            data = self.get_formatted_data(cache_key, ds_config, eval_config)

            responder_cls = get_responder_class_map().get(scenario_type)
            # sanity check
            if not responder_cls:
                raise ValueError('Unable to determine responder class')
            
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
        metrics = [config for config in metric_configs if scenario in config.scenarios]
        return metrics

    @staticmethod
    def matches_tag(config, tag):
        if not tag or not hasattr(config, 'tags'):
            return True
        config_tags = config.tags or []
        if tag in config_tags:
            return True
        return False

@click.command()
@click.option("--dataset", default=None, help="Only run datasets matching this tag")
@click.option("--scenario", default=None, help="Only run scenarios matching this tag")
@click.option("--model", default=None, help="Only run models matching this tag")
def run_eval(scenario, model, dataset):
    """Run evaluations for models using a judge model."""
    cli = EvalCommandLine()
    config_definitions = get_config_definitions()
    configs = load_and_validate_configs(config_definitions)

    model_tag = model
    dataset_tag = dataset
    scenario_tag = scenario

    eval_config = configs['eval']
    dataset_configs = configs['datasets']
    system_config = configs['system']
    metric_configs = configs['metrics']

    for model in eval_config.evaluated_models:
        if not EvalCommandLine.matches_tag(model, model_tag):
            click.echo(f'Skipping model {model.name} - does not match tag')
            continue
        # use the model specific values if they exist
        # else use the general eval values
        model.temperature = model.temperature or eval_config.temperature
        model.max_tokens = model.max_tokens or eval_config.max_tokens
        model.context_char_limit = model.context_char_limit or eval_config.context_char_limit

        for scenario in eval_config.scenarios:
            if not EvalCommandLine.matches_tag(scenario, scenario_tag):
                click.echo(f'Skipping scenario {scenario.type} - does not match tag')
                continue
            metrics = cli.get_metrics_for_scenario(scenario.type, metric_configs)
            prompt_builder = PromptBuilder(scenario.type, metrics)

            for dataset_name in scenario.datasets:
                ds_config = get_dataset_config(dataset_name, dataset_configs)
                if not EvalCommandLine.matches_tag(ds_config, dataset_tag):
                    click.echo(f'Skipping dataset {ds_config.name} - does not match tag')
                    continue
                cache_key = cli.cache.build_cache_key(dataset_name, scenario.type)

                eval_prompts = cli.get_evaluation_prompts(
                    cache_key,
                    model,
                    prompt_builder,
                    ds_config,
                    eval_config,
                    scenario.type
                )

                evaluation_results = cli.get_evaluation_results(
                    eval_prompts,
                    cache_key,
                    model,
                    system_config,
                    metrics
                )

                all_results = []
                for prompt, result in zip(eval_prompts, evaluation_results):
                    all_results.append({
                        **prompt,
                        **result
                    })
                save_evaluation_results(model.name, dataset_name, all_results)
