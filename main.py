from gpt_eval.evaluation import Evaluator
from gpt_eval.data.loader import load_formatted_data
from gpt_eval.config import (
    EVAL_CONFIG_PATH, 
    DATASET_CONFIG_PATH, 
    SYSTEM_CONFIG_PATH,
    SystemConfig, 
    EvaluationConfig, 
    DatasetConfig, 
    load_validated_config,
    RESPONDER_CLASS_MAP
)

from dotenv import load_dotenv
import os

load_dotenv()


def check_scenarios_valid_for_dataset(eval_config, datasets_config):
    for scenario in eval_config.scenarios:
        for dataset_name in scenario.datasets:
            dataset_config = next((config for config in datasets_config if config.name == dataset_name), None)
            if not dataset_config:
                raise ValueError(f"Dataset '{dataset_name}' is not configured.")
            if scenario.type not in dataset_config.scenarios:
                raise ValueError(f"Scenario type '{scenario.type}' is not valid for dataset '{dataset_name}'.")


if __name__ == "__main__":
    try:
        system_config = load_validated_config(SYSTEM_CONFIG_PATH, SystemConfig)
        eval_config = load_validated_config(EVAL_CONFIG_PATH, EvaluationConfig)
        dataset_configs = load_validated_config(DATASET_CONFIG_PATH, DatasetConfig, is_list=True)
        check_scenarios_valid_for_dataset(eval_config, dataset_configs)
    except Exception as e:
        print('Error while validating config')
        print(e)
        exit(1)


    for model in eval_config.evaluated_models:
        temperature = model.temperature or eval_config.temperature
        max_tokens = model.max_tokens or eval_config.max_tokens
        context_char_limit = model.context_char_limit or eval_config.context_char_limit
        for scenario in eval_config.scenarios:
            for dataset in scenario.datasets:
                filtered_ds_configs = filter(lambda ds: ds.name == dataset, dataset_configs)
                ds_config = next(filtered_ds_configs, None)
                # sanity check
                if not ds_config:
                    raise ValueError('Unable to determine dataset config')
                
                data = load_formatted_data(ds_config, eval_config.num_evals, eval_config.random_seed)
                
                responder_cls = RESPONDER_CLASS_MAP.get(scenario.type)
                # sanity check
                if not responder_cls:
                    raise ValueError('Unable to determine responder class')
                
                responder = responder_cls(
                    data=data,
                    api_type=model.api_type,
                    api_base=str(model.api_base), # pydantic httpurl field - must be coerced into string
                    temperature=temperature,
                    max_tokens=max_tokens,
                    context_char_limit=context_char_limit
                )
                eval_prompts = responder.get_evaluation_prompts()

                # necessary due to the way eval_prompts are collected
                # and how they are used by the evaluator class
                # this is a dirty work around until sqlite caching is implemented
                eval_prompts = [a['eval_prompt'] for a in eval_prompts]
                evaluator = Evaluator(
                    model_name=model.name,
                    dataset_name=ds_config.name,
                    evaluator_api_key=system_config.judge_api_key or os.getenv('OPENAI_KEY'),
                    evaluator_model=system_config.judge,
                    use_proxy=system_config.use_proxy,
                    proxies=system_config.proxies
                )
                evaluator.run_evaluation(eval_prompts)