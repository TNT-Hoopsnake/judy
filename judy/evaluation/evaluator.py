import os
import re
from typing import List, Tuple
from openai import AsyncOpenAI
from judy.utils import Retry
from judy.utils.prompts import SYSTEM_PROMPT
from judy.config import (
    RunConfig,
    MetricConfig,
    DEFAULT_OPENAI_API_BASE,
    get_est_token_cost,
)
from judy.responders import (
    MetricScore,
    EvalResponse,
    EvalPrompt,
)
from judy.config.logging import logger as log


class Evaluator:
    def __init__(self, run_config: RunConfig, metrics: List[MetricConfig]):
        """
        Initialize the Evaluator with the provided run configuration and metrics.

        Args:
            run_config (RunConfig): Configuration for the evaluation run.
            metrics (List[MetricConfig]): List of metrics to be used for evaluation.

        This class is responsible for interacting with the OpenAI API for model evaluations.
        It is initialized with the necessary configuration, including API key, model details,
        and evaluation parameters.
        """
        self.judge_client = AsyncOpenAI()
        self.judge_client.api_key = run_config.judge_api_key or os.getenv("OPENAI_KEY")
        # ensure the openai api_base points to the correct address
        # this can be updated by responders so it is necessary to set it here
        self.judge_client.api_base = DEFAULT_OPENAI_API_BASE
        if run_config.use_proxy and run_config.proxies:
            self.judge_client.proxy = {
                "http": str(run_config.proxies.http),
                "https": str(run_config.proxies.https),
            }
            log.debug("Proxy will be used for accessing openai API")

        self.metrics = metrics
        self.evaluator = run_config.judge
        self.evaluator_temperature = run_config.judge_temperature
        self.eval_input_tokens: List[int] = []
        self.eval_output_tokens: List[int] = []

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

        # Create a chat completion using the specified model and prompt
        completion = await self.judge_client.chat.completions.create(
            model=self.evaluator,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.evaluator_temperature,
        )
        log.info(
            "Estimated cost for single request to openai API: $%f",
            get_est_token_cost(
                self.evaluator,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
            ),
        )
        self.eval_input_tokens.append(completion.usage.prompt_tokens)
        self.eval_output_tokens.append(completion.usage.completion_tokens)
        # Extract and return the content of the generated evaluation response
        return completion.choices[0].message.content

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

        est_cost = get_est_token_cost(
            self.evaluator, sum(self.eval_input_tokens), sum(self.eval_output_tokens)
        )
        log.info("Total estimated cost of evaluation: $%f", est_cost)
        # update the cli progress bar when an evaluation result is completed
        progress_bar.update(1)
        return results
