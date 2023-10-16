from gpt_eval.evaluation import Evaluator
from gpt_eval.data.loader import load_formatted_data
from gpt_eval.config import (
    get_responder_class_map,
    load_and_validate_configs,
    get_config_definitions
)
from gpt_eval.cache import build_cache_key, get_cache, set_cache, calculate_content_hash
from gpt_eval.utils import save_evaluation_results, get_dataset_config, get_dataset_config
from dotenv import load_dotenv
import os

load_dotenv()


def get_evaluation_results(eval_prompts, cache_key, model, system_config):
    eval_results_cache_key = f"{calculate_content_hash(eval_prompts)}-{model.name}"
    if eval_results := get_cache(cache_key, eval_results_cache_key):
        print('Evaluation results retrieved from cache')
    else:
        print('Evaluation results not present in cache')
        evaluator = Evaluator(
            evaluator_api_key=system_config.judge_api_key or os.getenv('OPENAI_KEY'),
            evaluator_model=system_config.judge,
            use_proxy=system_config.use_proxy,
            proxies=system_config.proxies
        )
        eval_results = evaluator.run_evaluation(eval_prompts)
        set_cache(cache_key, eval_results_cache_key, eval_results)

    return eval_results


def get_formatted_data(cache_key, ds_config, eval_config):
    if data := get_cache(cache_key, 'data'):
        print('Formatted data retrieved from cache')
    else:
        print('Formatted data not present in cache')

        data = load_formatted_data(ds_config, eval_config.num_evals, eval_config.random_seed)
        set_cache(cache_key, 'data', data)

    return data


def get_evaluation_prompts(cache_key, model, ds_config, eval_config, scenario_type):
    eval_prompts_cache_key = f'eval_prompts-{model.name}'

    if eval_prompts := get_cache(cache_key, eval_prompts_cache_key):
        print('Evaluation prompts retrieved from cache')
    
    else:
        print('Evaluation prompts not present in cache')
        
        data = get_formatted_data(cache_key, ds_config, eval_config)

        responder_cls = get_responder_class_map().get(scenario_type)
        # sanity check
        if not responder_cls:
            raise ValueError('Unable to determine responder class')
        
        responder = responder_cls(
            data=data,
            api_type=model.api_type,
            api_base=str(model.api_base), # pydantic httpurl field - must be coerced into string
            temperature=model.temperature,
            max_tokens=model.max_tokens,
            context_char_limit=model.context_char_limit
        )
        eval_prompts = responder.get_evaluation_prompts()

        set_cache(cache_key, eval_prompts_cache_key, eval_prompts)

    return eval_prompts


if __name__ == "__main__":
    config_definitions = get_config_definitions()
    configs = load_and_validate_configs(config_definitions)

    eval_config = configs['eval']
    dataset_configs = configs['datasets']
    system_config = configs['system']

    for model in eval_config.evaluated_models:
        # use the model specific values if they exist
        # else use the general eval values
        model.temperature = model.temperature or eval_config.temperature
        model.max_tokens = model.max_tokens or eval_config.max_tokens
        model.context_char_limit = model.context_char_limit or eval_config.context_char_limit

        for scenario in eval_config.scenarios:
            for dataset_name in scenario.datasets:
                ds_config = get_dataset_config(dataset_name, dataset_configs)

                cache_key = build_cache_key(dataset_name, scenario.type)

                eval_prompts = get_evaluation_prompts(
                    cache_key,
                    model,
                    ds_config,
                    eval_config,
                    scenario.type
                )

                evaluation_results = get_evaluation_results(
                    eval_prompts,
                    cache_key,
                    model,
                    system_config
                )

                save_evaluation_results(model.name, dataset_name, evaluation_results)
