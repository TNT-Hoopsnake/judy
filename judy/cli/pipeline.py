import asyncio
import pathlib
from typing import List, Tuple
from tqdm.asyncio import tqdm
from judy.config import (
    dump_configs,
    DatasetConfig,
    RunConfig,
)
from judy.responders import (
    get_responder_class_map,
    EvalPrompt,
    EvalResponse,
)
from judy.utils import (
    PromptBuilder,
    dump_metadata,
)
from judy.utils.utils import ensure_directory_exists
from judy.config.settings import MODEL_BATCH_SIZE, JUDGE_BATCH_SIZE
from judy.config.logging import logger as log


class EvaluationPipeline:
    def __init__(self, manager):
        self.manager = manager
        self.stages = [
            asyncio.Queue(MODEL_BATCH_SIZE),
            asyncio.Queue(JUDGE_BATCH_SIZE),
            asyncio.Queue(),
        ]
        self.exception_queue = asyncio.Queue()
        self.cancel_event = asyncio.Event()
        self.results = {}

    async def run(self, models_to_run, evaluations_to_run):
        """Run all stages in this pipeline."""
        # Load data from datasets to construct prompts
        data_load_tasks = self.load_data_stage(
            self.stages[0], models_to_run, evaluations_to_run
        )

        # Get model responses for each prompt
        prompt_task = asyncio.create_task(
            self.model_prompt_stage(
                self.stages[0],
                self.stages[1],
                self.exception_queue,
                self.cancel_event,
                self.manager.pbar_model_prompting,
            )
        )

        # Get evaluation results for each model response
        evaluation_task = asyncio.create_task(
            self.evaluate_response_stage(
                self.stages[1],
                self.stages[2],
                self.exception_queue,
                self.cancel_event,
                self.manager.pbar_judge_prompting,
            )
        )

        # Save evaluation results to disk
        save_task = asyncio.create_task(
            self.save_results_stage(
                self.stages[2], self.exception_queue, self.cancel_event
            )
        )

        # Create a task to gather exceptions raised throughut the pipleine
        collect_exceptions = asyncio.create_task(
            self.collect_exceptions(self.exception_queue, self.cancel_event)
        )

        # Wait for all data to be loaded
        for task in asyncio.as_completed(data_load_tasks):
            await task
        self.manager.pbar_data_loading.close()

        # Wait for all stages to complete
        for stage in self.stages:
            await stage.join()
        await self.exception_queue.join()

        # Cancel tasks blocked waiting on empty queues
        self.cancel_event.set()
        for task in [prompt_task, evaluation_task, collect_exceptions, save_task]:
            task.cancel()
        self.manager.pbar_model_prompting.close()
        self.manager.pbar_judge_prompting.close()

        # Wait for evaluation results to be saved
        for task in asyncio.as_completed(self.manager.evaluation_results):
            await task

    async def collect_exceptions(
        self,
        exc_queue: asyncio.Queue,
        cancel_event: asyncio.Event,
    ):
        """Collects exceptions from the exception queue."""
        while not cancel_event.is_set():
            exc = await exc_queue.get()
            self.manager.exceptions.append(exc)
            exc_queue.task_done()

    def load_data_stage(self, queue, models_to_run, evaluations_to_run):
        """A producer which schedules tasks to load data from datasets for each model."""
        data_load_tasks = []
        # Extract prompts from datasets for each model
        for eval_model in models_to_run:
            log.info("Evaluation started for model: %s", eval_model.id)
            # Model-specific parameters override general run config parameters
            eval_model.temperature = (
                eval_model.temperature or self.manager.run_config.temperature
            )
            eval_model.max_tokens = (
                eval_model.max_tokens or self.manager.run_config.max_tokens
            )
            eval_model.context_char_limit = (
                eval_model.context_char_limit
                or self.manager.run_config.context_char_limit
            )

            # create a subdirectory to save the results for the model currently being evaluated
            # clear any existing results saved for this model
            model_results_dir = ensure_directory_exists(
                self.manager.results_dir / eval_model.id, clear_if_exists=True
            )
            # dump configurations and metadata settings per eval model
            # this allows cumulative results with different configs, under the same run name, but with different models
            dump_configs(model_results_dir, self.manager.configs)
            dump_metadata(
                model_results_dir,
                self.manager.dataset_tag,
                self.manager.task_tag,
                self.manager.model_tag,
                eval_model.id,
            )

            ensure_directory_exists(model_results_dir / "datasets")

            for scenario_id, dataset_id, task_id in evaluations_to_run:
                ds_config = self.manager.config_cache["datasets"].get(dataset_id)
                scenario_metrics = self.manager.config_cache["scenario_metrics"].get(
                    scenario_id
                )
                prompt_builder = PromptBuilder(task_id, scenario_metrics)

                cache_key = self.manager.cache.build_cache_key(dataset_id, task_id)
                eval_id = (
                    scenario_id,
                    dataset_id,
                    task_id,
                )
                log.info(
                    "Queueing scenario: %s, dataset: %s, task: %s",
                    scenario_id,
                    dataset_id,
                    task_id,
                )
                format_data_task = asyncio.create_task(
                    self.format_data_stage(
                        queue,
                        cache_key,
                        eval_model,
                        prompt_builder,
                        ds_config,
                        self.manager.run_config,
                        eval_id,
                        self.manager.pbar_data_loading,
                    )
                )
                data_load_tasks.append(format_data_task)
        return data_load_tasks

    async def format_data_stage(
        self,
        queue: asyncio.Queue,
        cache_key: str,
        model: str,
        prompt_builder: PromptBuilder,
        ds_config: DatasetConfig,
        run_config: RunConfig,
        eval_ids: Tuple[str, str, str],
        progress_bar: tqdm,
    ):
        """
        Asynchronously retrieves responses for prompts for a given model, dataset, task and run configuration
        It first checks the cache for existing responses and if found, returns them.
        If not found or if caching is ignored, the relevant Responder class is used to generate the prompts using the provided parameters.
        The generated prompts are cached before being returned

        Args:
            queue (asyncio.Queue): Queue for storing the generated prompts for later retrieval.
            cache_key (str): Cache key for storing/retrieving prompts.
            model (str): Model identifier.
            prompt_builder (PromptBuilder): PromptBuilder instance.
            ds_config (DatasetConfig): Dataset configuration.
            run_config (RunConfig): Run configuration.
            progress_bar (tqdm): Progress bar for tracking the number of prompts generated.
        """
        task_id = eval_ids[2]
        responder_cls = get_responder_class_map().get(task_id)
        # sanity check
        if not responder_cls:
            log.error("Unable to determine responder class for task: %s", task_id)
            raise ValueError("Unable to determine responder class")

        async for item in self.manager.generate_formatted_data(
            cache_key, ds_config, run_config, task_id
        ):
            responder = responder_cls(
                data=item,
                prompt_builder=prompt_builder,
                model_config=model,
            )
            progress_bar.update(1)
            await queue.put((responder, cache_key, eval_ids))

    async def model_prompt_stage(
        self,
        in_queue: asyncio.Queue,
        out_queue: asyncio.Queue,
        exc_queue: asyncio.Queue,
        cancel_event: asyncio.Event,
        progress_bar: tqdm,
    ):
        """
        Greedily consumes formatted data from the queue and schedules tasks to prompt models with them.

        It first checks the cache for existing responses and if found, returns them. If not found or if caching is ignored, the relevant Responder class is used to generate the prompts using the provided parameters.
        The generated prompts are cached before being returned"""
        while not cancel_event.is_set():
            try:
                responder, cache_key, eval_ids = await in_queue.get()
                model_prompts = await responder.get_model_prompts()
                model_response = None
                model_response_cache_key = f"model_response-{self.manager.cache.calculate_content_hash(model_prompts)}-{responder.model_id}"
                if self.manager.ignore_prompt_cache:
                    log.info("Skipped accessing cache for model responses")
                else:
                    model_response = self.manager.cache.get(
                        cache_key, model_response_cache_key
                    )

                data = None
                if model_response:
                    log.info("Model response retrieved from cache")
                    data = model_response
                else:
                    log.info("Model response not present in cache")
                    model_task = asyncio.create_task(
                        responder.get_responses_for_prompts(
                            model_prompts,
                            progress_bar,
                        )
                    )
                    data = model_task
                await out_queue.put(
                    (
                        data,
                        responder,
                        cache_key,
                        model_response_cache_key,
                        eval_ids,
                    )
                )
            except Exception as exc:
                await exc_queue.put(exc)
            finally:
                in_queue.task_done()

    async def evaluate_response_stage(
        self,
        in_queue: asyncio.Queue,
        out_queue: asyncio.Queue,
        exc_queue: asyncio.Queue,
        cancel_event: asyncio.Event,
        progress_bar,
    ):
        """Greedily consumes model prompt tasks from the queue and schedules tasks to run evaluations for them."""
        while not cancel_event.is_set():
            (
                response,
                responder,
                cache_key,
                model_response_cache_key,
                eval_ids,
            ) = await in_queue.get()
            try:
                await out_queue.put(
                    (
                        asyncio.create_task(
                            self.manager.run_evaluation(
                                response,
                                responder,
                                cache_key,
                                model_response_cache_key,
                                progress_bar,
                                exc_queue,
                            )
                        ),
                        responder.model_id,
                        eval_ids,
                    )
                )
            except Exception as exc:
                await exc_queue.put(exc)
            finally:
                in_queue.task_done()

    async def save_results_stage(
        self,
        queue: asyncio.Queue,
        exc_queue: asyncio.Queue,
        cancel_event: asyncio.Event,
    ):
        """
        Greedily consume evaluation result tasks and schedules tasks to write results to a JSON file.

        Args:
            results_dir (str | pathlib.Path): The directory to save the evaluation results in.
            scenario_id (str): The identifier for the evaluation scenario.
            task_id (str): The identifier for the evaluated task.
            dataset_id (str): The name of the evaluated dataset.
            eval_prompts (List[EvalPrompt]): List of evaluation prompts.
            eval_results (List[EvalResponse]): List of evaluation results.

        The function creates a directory structure based on the model name and saves the
        evaluation results for each dataset in their own JSON file. Each entry in the JSON file corresponds to a
        specific evaluation prompt and its corresponding response.

        The structure of the saved JSON file is as follows:
        [{
            "dataset_id": "dataset_id",
            "task_id": "task_id",
            "scenario_id": "scenario_id",
            "model": {
                "response": "response",
                "other_model_data": ...
            },
            "evaluator": {
                "evaluator_data": ...
            }
        }]
        """
        while not cancel_event.is_set():
            try:
                eval_task, model_id, eval_ids = await queue.get()
                results_dir = self.manager.get_model_dataset_dir(model_id)
                scenario_id, dataset_id, task_id = eval_ids
                eval_data = await eval_task
                eval_prompts: List[EvalPrompt] = eval_data[0]
                eval_results: List[EvalResponse] = eval_data[1]
                clean_ds_name = dataset_id.split("/")[-1]
                data = []
                for idx, item in enumerate(eval_results):
                    model = {
                        "response": eval_prompts[idx].response_data.response,
                        **eval_prompts[idx].response_data.prompt.model_dump(),
                    }
                    data.append(
                        {
                            "dataset_id": dataset_id,
                            "task_id": task_id,
                            "scenario_id": scenario_id,
                            "model": model,
                            "evaluator": item.model_dump(mode="json"),
                        }
                    )
                    key = f"{clean_ds_name}-{scenario_id}-{task_id}"
                    if key in self.results:
                        self.results[key] = self.results[key] + 1
                    else:
                        self.results[key] = 1

                self.manager.evaluation_results.append(
                    asyncio.create_task(
                        self.manager.save_result(
                            data,
                            pathlib.Path(results_dir)
                            / f"{clean_ds_name}-{scenario_id}-{task_id}.json",
                            exc_queue,
                        )
                    )
                )
            except Exception as exc:
                await exc_queue.put(exc)
            finally:
                queue.task_done()
