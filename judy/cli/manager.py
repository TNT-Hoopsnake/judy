import pathlib
from typing import List, Optional
import sys

from dotenv import load_dotenv

from judy.cache import SqliteCache
from judy.cli.install import setup_user_dir
from judy.config import (
    get_dataset_config,
    get_task_config,
    DatasetConfig,
    TaskTypes,
    ScenarioConfig,
    EvaluationConfig,
    MetricConfig,
    RunConfig,
    IgnoreCacheTypes,
)
from judy.config.data_models import EvaluatedModel
from judy.dataset import BaseFormattedData
from judy.dataset.loader import load_formatted_data
from judy.evaluation import Evaluator
from judy.responders import get_responder_class_map
from judy.utils import (
    PromptBuilder,
    matches_tag,
)

from judy.config.logging import logger as log


class EvalManager:
    def __init__(
        self, config_paths: List[str | pathlib.Path] = None, clear_cache: bool = False
    ):
        load_dotenv()
        setup_user_dir()
        self.cache = SqliteCache(config_paths, clear_cache)

    def sizeof_current_run_cache(self):
        return len(self.cache.cache)

    def get_num_cached_runs(self):
        tables = self.cache.cache.get_tablenames(self.cache.cache.filename)
        return len(tables)

    def get_evaluation_results(
        self,
        eval_prompts: List[dict],
        cache_key: str,
        model: str,
        run_config: RunConfig,
        metrics: List[MetricConfig],
        ignore_cache_type: IgnoreCacheTypes,
    ):
        eval_results = None
        eval_results_cache_key = (
            f"{self.cache.calculate_content_hash(eval_prompts)}-{model.id}"
        )
        if ignore_cache_type and ignore_cache_type in [
            IgnoreCacheTypes.ALL,
            IgnoreCacheTypes.PROMPTS,
        ]:
            log.info("Skipped accessing cache for evaluation results")
        else:
            eval_results = self.cache.get(cache_key, eval_results_cache_key)
            if not eval_results:
                log.info("Evaluation results not present in cache")

        if eval_results:
            log.info("Evaluation results retrieved from cache")
        else:
            evaluator = Evaluator(run_config=run_config, metrics=metrics)
            eval_results = evaluator.run_evaluation(eval_prompts)
            self.cache.set(cache_key, eval_results_cache_key, eval_results)

        return eval_results

    def get_formatted_data(
        self,
        cache_key: str,
        ds_config: DatasetConfig,
        run_config: RunConfig,
        ignore_cache: bool,
    ) -> BaseFormattedData:
        data = None
        if ignore_cache:
            log.info("Skipped accessing cache for formatted data")
        else:
            data = self.cache.get(cache_key, "data")
            if not data:
                log.info("Formatted data not present in cache")

        if data:
            log.info("Formatted data retrieved from cache")

        else:
            try:
                data = load_formatted_data(
                    ds_config,
                    run_config.num_evals,
                    run_config.random_seed,
                    ignore_cache,
                )
                self.cache.set(cache_key, "data", data)
            except Exception as e:
                log.error(str(e))
                sys.exit(1)

        return data

    def get_evaluation_prompts(
        self,
        cache_key: str,
        model: str,
        prompt_builder: PromptBuilder,
        ds_config: DatasetConfig,
        run_config: RunConfig,
        task_type: TaskTypes,
        ignore_cache_type: IgnoreCacheTypes,
    ) -> List[dict]:
        eval_prompts_cache_key = f"eval_prompts-{model.id}"
        eval_prompts = None
        if ignore_cache_type and ignore_cache_type in [
            IgnoreCacheTypes.ALL,
            IgnoreCacheTypes.PROMPTS,
        ]:
            log.info("Skipped accessing cache for evaluation prompts")
        else:
            eval_prompts = self.cache.get(cache_key, eval_prompts_cache_key)
            if not eval_prompts:
                log.info("Evaluation prompts not present in cache")

        if eval_prompts:
            log.info("Evaluation prompts retrieved from cache")
        else:
            ignore_dataset_cache: bool = ignore_cache_type in [
                IgnoreCacheTypes.ALL,
                IgnoreCacheTypes.DATASET,
            ]
            data = self.get_formatted_data(
                cache_key, ds_config, run_config, ignore_dataset_cache
            )

            responder_cls = get_responder_class_map().get(task_type)
            # sanity check
            if not responder_cls:
                log.error("Unable to determine responder class for task: %s", task_type)
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
    def collect_evaluations(
        run_config: RunConfig,
        eval_config: EvaluationConfig,
        dataset_config_list: List[DatasetConfig],
    ):
        evaluations_to_run = []
        config_cache = {
            "scenario_metrics": {},
            "datasets": {},
        }
        scenarios_to_run: List[ScenarioConfig] = EvalManager.get_scenarios_for_run(
            run_config, eval_config
        )
        for eval_scenario in scenarios_to_run:
            metrics = EvalManager.get_metrics_for_scenario(eval_scenario)
            config_cache["scenario_metrics"].setdefault(eval_scenario.id, metrics)
            for dataset_id in eval_scenario.datasets:
                dataset = get_dataset_config(dataset_id, dataset_config_list)
                for task_id in dataset.tasks:
                    task = get_task_config(task_id, eval_config)
                    evaluations_to_run.append((eval_scenario.id, dataset.id, task.id))
                    config_cache["datasets"].setdefault(dataset.id, dataset)
        return evaluations_to_run, scenarios_to_run, config_cache

    @staticmethod
    def get_scenarios_for_run(
        run_config: RunConfig,
        eval_config: EvaluationConfig,
        task_tag: Optional[str] = None,
    ) -> List[ScenarioConfig]:
        scenarios = []
        for scenario_id in run_config.scenarios:
            matching_scenario = next(
                filter(
                    lambda scenario: scenario.id
                    == scenario_id,  # pylint: disable=cell-var-from-loop
                    eval_config.scenarios,
                )
            )
            # Ensure selected task is defined in the eval config
            assert (
                matching_scenario
            ), f"Scenario {scenario_id} is undefined. Create an entry for it in the evaluation config"
            if not matches_tag(matching_scenario, task_tag):
                log.warning(
                    "Skipping scenario (%s) as it does not match tag (%s)",
                    matching_scenario.id,
                    task_tag,
                )
                continue
            scenarios.append(matching_scenario)
        return scenarios

    @staticmethod
    def get_metrics_for_scenario(scenario: ScenarioConfig):
        metrics = []
        for metric_config in scenario.metrics:
            # override group values with metric specific values
            metric_config.score_min = metric_config.score_min or scenario.score_min
            metric_config.score_max = metric_config.score_max or scenario.score_max
            metrics.append(metric_config)
        return metrics

    @staticmethod
    def get_models_to_run(
        run_config: RunConfig, model_tag: Optional[str] = None
    ) -> List[EvaluatedModel]:
        models: List[EvaluatedModel] = []
        for model in run_config.models:
            if not matches_tag(model, model_tag):
                log.warning(
                    "Skipping model (%s) as it does not match tag (%s)",
                    model.id,
                    model_tag,
                )
                continue

            models.append(model)

        return models
