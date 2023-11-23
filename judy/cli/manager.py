import pathlib
from typing import List, Optional
from dotenv import load_dotenv

from judy.cache import SqliteCache
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
        """
        Initialize the EvalManager instance.

        Args:
            config_paths (List[str | pathlib.Path], optional): List of paths to configuration files. Defaults to None.
            clear_cache (bool, optional): Flag to clear the cache. Defaults to False.
        """
        load_dotenv()
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
        progress_bar,
    ):
        """
        Retrieves the evaluation results for a given set of evaluation prompts.
        It first checks the cache for existing results and, if found, returns them.
        If not found or if caching is ignored, it runs the evaluation using the provided parameters,
        caches the results, and returns them.

        Args:
            eval_prompts (List[dict]): List of evaluation prompts.
            cache_key (str): Cache key for storing/retrieving results.
            model (str): Model identifier.
            run_config (RunConfig): Run configuration.
            metrics (List[MetricConfig]): List of metric configurations.
            ignore_cache_type (IgnoreCacheTypes):` Type of cache to ignore.
            progress_bar (tdqm): Instance of tqdm progress bar, used to display updates on run state to the user


        Returns:
            Any: Evaluation results.
        """
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
            progress_bar.update(len(eval_results))
            log.info("Evaluation results retrieved from cache")
        else:
            evaluator = Evaluator(run_config=run_config, metrics=metrics)
            eval_results = evaluator.run_evaluation(eval_prompts, progress_bar)
            self.cache.set(cache_key, eval_results_cache_key, eval_results)

        return eval_results

    def get_formatted_data(
        self,
        cache_key: str,
        ds_config: DatasetConfig,
        run_config: RunConfig,
        task_type: TaskTypes,
        ignore_cache: bool,
    ) -> BaseFormattedData:
        """
        Retrieves formatted data for a given dataset and run configuration.
        It first checks the cache for existing data and, if found, returns it.
        If not found or if caching is ignored, it loads the formatted data, caches it, and returns it.

        Args:
            cache_key (str): Cache key for storing/retrieving data.
            ds_config (DatasetConfig): Dataset configuration.
            run_config (RunConfig): Run configuration.
            task_type (TaskTypes): ID of the task that will use the formatted data
            ignore_cache (bool): Flag to ignore cache.

        Returns:
            BaseFormattedData: Formatted data.
        """
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
                    task_type,
                    ignore_cache,
                )
                self.cache.set(cache_key, "data", data)
            except Exception as e:
                log.error(str(e))
                return None

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
        """
        Retrieves evaluation prompts for a given model, dataset, task and run configuration
        It first checks the cache for existing prompts and, if found, returns them.
        If not found or if caching is ignored, the relevant Responder class is used to generate the prompts using the provided parameters.
        The generated prompts are cached before being returned

        Args:
            cache_key (str): Cache key for storing/retrieving prompts.
            model (str): Model identifier.
            prompt_builder (PromptBuilder): PromptBuilder instance.
            ds_config (DatasetConfig): Dataset configuration.
            run_config (RunConfig): Run configuration.
            task_type (TaskTypes): Type of the task.
            ignore_cache_type (IgnoreCacheTypes): Type of cache to ignore.

        Returns:
            List[dict]: List of evaluation prompts.
        """
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
                cache_key, ds_config, run_config, task_type, ignore_dataset_cache
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
        dataset_tag: str = None,
        task_tag: str = None,
    ):
        """
        Collects evaluations to run, scenarios to run, and the metrics/datasets to be used in the evaluations,
        based on the provided run configuration, evaluation configuration, and a list of dataset configurations.
        It iterates through scenarios defined in the run configuration, matches them with the scenarios
        defined in the evaluation configuration, and builds a list of evaluations to run.

        Args:
            run_config (RunConfig): Run configuration.
            eval_config (EvaluationConfig): Evaluation configuration.
            dataset_config_list (List[DatasetConfig]): List of dataset configurations.
            dataset_tag (str): An optional tag for filtering datasets included in the evaluations
            task_tag (str): An optional tag for filtering tasks included in the evaluations

        Returns:
            Tuple[List[Tuple[str, str, str]], List[ScenarioConfig], dict]: Tuple containing evaluations to run,
            scenarios to run, and configuration cache.
        """
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
                if matches_tag(dataset, dataset_tag):
                    for ds_task in dataset.tasks:
                        task = get_task_config(ds_task.id, eval_config)
                        if matches_tag(task, task_tag):
                            evaluations_to_run.append(
                                (eval_scenario.id, dataset.id, task.id)
                            )
                            config_cache["datasets"].setdefault(dataset.id, dataset)
        return evaluations_to_run, scenarios_to_run, config_cache

    @staticmethod
    def get_scenarios_for_run(
        run_config: RunConfig,
        eval_config: EvaluationConfig,
        task_tag: Optional[str] = None,
    ) -> List[ScenarioConfig]:
        """
        Retrieves scenarios for a given run configuration and evaluation configuration,
        optionally filtered by a task tag.

        Args:
            run_config (RunConfig): Run configuration.
            eval_config (EvaluationConfig): Evaluation configuration.
            task_tag (Optional[str], optional): Task tag. Defaults to None.

        Returns:
            List[ScenarioConfig]: List of scenarios for the run.
        """
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
        """
        Retrieves metrics for a given scenario, overriding scenario values
        with metric-specific values if they exist.

        Args:
            scenario (ScenarioConfig): Scenario configuration.

        Returns:
            List[MetricConfig]: List of metric configurations for the scenario.
        """
        metrics = []
        for metric_config in scenario.metrics:
            # override scenario values with metric specific values
            metric_config.score_min = metric_config.score_min or scenario.score_min
            metric_config.score_max = metric_config.score_max or scenario.score_max
            metrics.append(metric_config)
        return metrics

    @staticmethod
    def get_models_to_run(
        run_config: RunConfig, model_tag: Optional[str] = None
    ) -> List[EvaluatedModel]:
        """
        Retrieves models to run based on the provided run configuration,
        optionally filtered by a model tag.

        Args:
            run_config (RunConfig): Run configuration.
            model_tag (Optional[str], optional): Model tag. Defaults to None.

        Returns:
            List[EvaluatedModel]: List of models to run.
        """
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
