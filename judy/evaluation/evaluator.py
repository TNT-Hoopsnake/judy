import re
from typing import List, Tuple
from judy.utils import Retry, LLM, LLMModel
from judy.evaluation.prompts import SYSTEM_PROMPT
from judy.config import RunConfig, MetricConfig
from judy.responders import (
    MetricScore,
    EvalResponse,
    EvalPrompt,
)
from judy.config.logging import logger as log


class Evaluator:
    def __init__(self, llm: LLM, run_config: RunConfig, metrics: List[MetricConfig]):
        """
        Initialize the Evaluator with the provided run configuration and metrics.

        Args:
            run_config (RunConfig): Configuration for the evaluation run.
            metrics (List[MetricConfig]): List of metrics to be used for evaluation.

        This class is responsible for interacting with the OpenAI API for model evaluations.
        It is initialized with the necessary configuration, including API key, model details,
        and evaluation parameters.
        """
        self.metrics = metrics
        self.evaluator_temperature = run_config.judge.temperature
        self.llm = llm
        self.llm_model = LLMModel(
            api_type=run_config.judge.api_type,
            api_base=run_config.judge.api_base,
            api_key=run_config.judge.api_key,
            name=run_config.judge.name,
        )

    async def query_model(self, prompt: str):
        """Query the model with the given prompt."""
        return await self.llm.query_chat_model(self.llm_model, prompt)

    @Retry()
    async def get_evaluation_response(self, prompt: str) -> str:
        """
        Sends a request to the OpenAI API to obtain an evaluation response
        based on the provided prompt and model details.

        Args:
            prompt (str): The text prompt to be used for generating an evaluation.

        Returns:
            str: The generated evaluation response.
        """

        return await self.llm.complete(
            self.llm_model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.evaluator_temperature,
        )

    def parse_result(self, result: str) -> Tuple[int]:
        """
        Extracts metric scores from the evaluation result string and returns
        a tuple of scores based on the defined metrics.

        Args:
            result (str): The input string containing metric scores.

        Returns:
            tuple: A tuple containing scores based on the defined criteria.
        """
        # Initialize variables to store scores
        scores = [0] * len(self.metrics)

        # Iterate through criteria and extract their scores
        for i, metric in enumerate(self.metrics):
            # Search for the metric name and its associated score
            m = re.search(rf"{metric.name}: ([\d]+)", result, re.IGNORECASE)

            # If the metric is found, update the corresponding variable with its score
            if m is not None:
                score = m.group(1)
                try:
                    score = int(score)
                    # scoring method can be changed arbitrarily
                    # eg: user might want to assign 1 as the highest score
                    # we still need to ensure the evaluated score is within range
                    min_score = min(metric.score_min, metric.score_max)
                    max_score = max(metric.score_min, metric.score_max)
                    if not min_score <= score <= max_score:
                        raise ValueError(
                            f"Evaluated score was out of range: {score} - should be between {min_score} and {max_score}"
                        )
                    scores[i] = score
                except ValueError as e:
                    log.warning(str(e))
                    scores[i] = 0

        return tuple(scores)

    async def get_evaluation_results(self, prompt: str) -> EvalResponse:
        """
        Generates evaluation results for a single prompt, including the prompt,
        generated response, and associated metric scores.

        Args:
            prompt (str): The text prompt for evaluation.

        Returns:
            EvalResponse: Evaluation response containing prompt, generated response, and metric scores.
        """
        eval_result = await self.get_evaluation_response(prompt)

        # Parse the evaluation metrics from the result using the criteria dictionary
        metric_scores = self.parse_result(eval_result)
        # Dynamically populate the model_result_dict with scores for each criterion
        scores = []
        for i, metric in enumerate(self.metrics):
            scores.append(MetricScore(name=metric.name, score=metric_scores[i]))

        return EvalResponse(prompt=prompt, response=eval_result, scores=scores)

    async def run_evaluation(self, prompt: EvalPrompt, progress_bar) -> EvalResponse:
        """
        Runs evaluations for a list of prompts and returns a list of evaluation responses,
        including prompts, generated responses, and associated metric scores.

        Args:
            prompts (List[EvalPrompt]): List of prompts for evaluation.
            progress_bar (tdqm): Instance of tqdm progress bar, used to display updates on run state to the user

        Returns:
            List[EvalResponse]: List of evaluation responses.
        """
        results = await self.get_evaluation_results(prompt.prompt)

        progress_bar.update(1)
        return results
