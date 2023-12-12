import json
import asyncio
import pathlib
from typing import List, Optional, Dict
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from pydantic import BaseModel
import numpy as np

from judy.cache import SqliteCache
from judy.cli.pipeline import EvaluationPipeline
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
from judy.dataset.loader import load_formatted_data, get_dataset, get_eval_idxs
from judy.evaluation import Evaluator
from judy.responders import (
    BaseResponder,
    ModelResponse,
)
from judy.utils import (
    matches_tag,
)

from judy.config.logging import logger as log


class EvalManager:
    def __init__(
        self,
        tags: Dict[str, str],
        configs: Dict[str, BaseModel],
        results_dir: str | pathlib.Path,
        config_paths: List[str | pathlib.Path] = None,
        clear_cache: bool = False,
        ignore_cache: IgnoreCacheTypes = None,
    ):
        """
        Initialize the EvalManager instance.

        Args:
            config_paths (List[str | pathlib.Path], optional): List of paths to configuration files. Defaults to None.
            clear_cache (bool, optional): Flag to clear the cache. Defaults to False.
        """
        load_dotenv()
        self.cache = SqliteCache(config_paths, clear_cache)
        self.results_dir = results_dir
        self.evaluation_results = []
        self.configs = configs
        self.eval_config = configs["eval"]
        self.dataset_config = configs["datasets"]
        self.run_config = configs["run"]

        self.model_tag = tags["model"]
        self.dataset_tag = tags["dataset"]
        self.task_tag = tags["task"]

        self.config_cache = {
            "scenario_metrics": {},
            "datasets": {},
        }

        self.pbar_data_loading = None
        self.pbar_model_prompting = None
        self.pbar_judge_prompting = None

        self.ignore_cache = ignore_cache
        self.ignore_dataset_cache: bool = self.ignore_cache in [
            IgnoreCacheTypes.ALL,
            IgnoreCacheTypes.DATASET,
        ]
        self.ignore_prompt_cache: bool = self.ignore_cache in [
            IgnoreCacheTypes.ALL,
            IgnoreCacheTypes.PROMPTS,
        ]
        self.ignore_eval_cache: bool = self.ignore_cache in [
            IgnoreCacheTypes.ALL,
            IgnoreCacheTypes.EVALS,
        ]

        self.exceptions = []

        self.pipeline = EvaluationPipeline(self)

    async def run_pipeline(self, models_to_run, evaluations_to_run):
        await self.pipeline.run(models_to_run, evaluations_to_run)

    async def generate_formatted_data(
        self,
        cache_key: str,
        ds_config: DatasetConfig,
        run_config: RunConfig,
        task_type: TaskTypes,
    ) -> BaseFormattedData:
        """
        Generates formatted data for a given dataset and run configuration.
        It first checks the cache for existing data and, if found, returns it.
        If not found or if caching is ignored, it loads the formatted data, caches it, and returns it.

        Args:
            cache_key (str): Cache key for storing/retrieving data.
            ds_config (DatasetConfig): Dataset configuration.
            run_config (RunConfig): Run configuration.
            task_type: (TaskTypes): The id of the task used to generate the data.

        Returns:
            BaseFormattedData: Formatted data.
        """
        data = None
        if self.ignore_dataset_cache:
            log.info("Skipped accessing cache for formatted data")
        else:
            data = self.cache.get(cache_key, "data")
            if not data:
                log.info("Formatted data not present in cache")

        if data:
            log.info("Formatted data retrieved from cache")
            for item in data:
                yield item
        else:
            try:
                np.random.seed(run_config.random_seed)
                dataset = await get_dataset(ds_config, self.ignore_dataset_cache)
                eval_idxs = get_eval_idxs(run_config.num_evals, len(dataset))
                cache_entry = []
                for eval_idx in eval_idxs:
                    data = load_formatted_data(
                        dataset,
                        ds_config,
                        eval_idx,
                        task_type,
                    )
                    cache_entry.append(data)
                    yield data
                self.cache.set(cache_key, "data", cache_entry)
            except Exception as e:
                log.error(str(e))

    async def run_evaluation(
        self,
        response: asyncio.Task | List[ModelResponse],
        responder: BaseResponder,
        cache_key: str,
        model_response_cache_key: str,
        progress_bar,
        exc_queue: asyncio.Queue,
    ):
        """
        Asynchronously retrieves the evaluation results for an evaluation prompt.

        It first checks the cache for existing results and, if found, returns them. If not found or if caching is ignored, it runs the evaluation using the provided parameters, caches the results, and returns them.
        """
        try:
            eval_results = []
            eval_prompts = []
            if isinstance(response, asyncio.Task):
                model_responses = await response
                self.cache.set(cache_key, model_response_cache_key, model_responses)
            else:
                model_responses = response
            for model_response in model_responses:
                eval_prompt = await responder.build_eval_prompt(model_response)
                eval_prompts.append(eval_prompt)
                eval_results_cache_key = f"eval-{self.cache.calculate_content_hash(eval_prompt)}-{responder.model_id}"
                if self.ignore_eval_cache:
                    log.info("Skipped accessing cache for evaluation results")
                else:
                    eval_results = self.cache.get(cache_key, eval_results_cache_key)
                    if not eval_results:
                        log.info("Evaluation results not present in cache")

                if eval_results:
                    progress_bar.update(1)
                    log.info("Evaluation results retrieved from cache")
                else:
                    eval_results = []
                    evaluator = Evaluator(
                        run_config=self.run_config, metrics=responder.pb.metric_configs
                    )
                    eval_results.append(
                        await evaluator.run_evaluation(eval_prompt, progress_bar)
                    )
                    self.cache.set(cache_key, eval_results_cache_key, eval_results)
            return eval_prompts, eval_results
        except Exception as exc:
            await exc_queue.put(exc)

    async def save_result(self, data, filename: pathlib.Path):
        """Save the evaluation results to a file."""
        with open(
            filename,
            "w+",
            encoding="utf-8",
        ) as fn:
            json.dump(data, fn, indent=4)

    def collect_evaluations(
        self,
    ):
        """
        Collects evaluations to run, scenarios to run, and the metrics/datasets to be used in the evaluations.

        Collection is based on the provided run configuration, evaluation configuration, and a list of dataset configurations. It iterates through scenarios defined in the run configuration, matches them with the scenarios defined in the evaluation configuration, and builds a list of evaluations to run.
        """
        evaluations_to_run = []
        scenarios_to_run: List[ScenarioConfig] = EvalManager.get_scenarios_for_run(
            self.run_config, self.eval_config
        )
        for eval_scenario in scenarios_to_run:
            metrics = EvalManager.get_metrics_for_scenario(eval_scenario)
            self.config_cache["scenario_metrics"].setdefault(eval_scenario.id, metrics)
            for dataset_id in eval_scenario.datasets:
                dataset = get_dataset_config(dataset_id, self.dataset_config)
                if matches_tag(dataset, self.dataset_tag):
                    for ds_task in dataset.tasks:
                        task = get_task_config(ds_task.id, self.eval_config)
                        if matches_tag(task, self.task_tag):
                            evaluations_to_run.append(
                                (eval_scenario.id, dataset.id, task.id)
                            )
                            self.config_cache["datasets"].setdefault(
                                dataset.id, dataset
                            )
        return evaluations_to_run, scenarios_to_run

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
    def get_metrics_for_scenario(scenario: ScenarioConfig) -> List[MetricConfig]:
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

    def get_model_dataset_dir(self, model_id: str) -> pathlib.Path:
        """Get the directory for a model's datasets."""
        return pathlib.Path(self.results_dir / model_id) / "datasets"

    def sizeof_current_run_cache(self):
        """Get the size of the current run cache."""
        return len(self.cache.cache)

    def get_num_cached_runs(self):
        """Get the number of cached runs."""
        tables = self.cache.cache.get_tablenames(self.cache.cache.filename)
        return len(tables)

    def initialize_progress_bars(self, num_evals):
        """Initialize progress bars for data loading, model prompting, and evaluation."""
        self.pbar_data_loading = tqdm(total=num_evals, position=0, desc="Loading data")
        self.pbar_model_prompting = tqdm(
            total=num_evals, position=1, desc="Prompting models"
        )
        self.pbar_judge_prompting = tqdm(
            total=num_evals, position=2, desc="Evaluating Model Responses"
        )

    @staticmethod
    def update_progress(progress_bar, num_items=1):
        """Update a progress bar."""
        progress_bar.update(num_items)
