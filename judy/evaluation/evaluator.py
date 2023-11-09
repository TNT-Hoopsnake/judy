import os
import re
import logging
from typing import List, Tuple
import openai
from judy.utils import Retry
from judy.utils.prompts import SYSTEM_PROMPT
from judy.config.settings import JudgeModels
from judy.config import (
    RunConfig,
    MetricConfig,
)
from judy.responders import (
    MetricScore,
    EvalResponse,
    EvalPrompt,
)

_logger = logging.getLogger("app")


def get_est_token_cost(
    eval_model: JudgeModels, num_input_tokens: int, num_output_tokens
) -> float:
    input_cost = output_cost = 0
    match eval_model:
        case JudgeModels.GPT4:
            input_cost = 0.01 / 1000.0
            output_cost = 0.03 / 1000.0
        case JudgeModels.GPT35:
            input_cost = 0.001 / 1000.0
            output_cost = 0.002 / 1000.0
        case _:
            _logger.error(
                "Unable to determine cost for given judge model: %s", eval_model
            )

    return round((num_input_tokens * input_cost) + (num_output_tokens * output_cost), 5)


DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"


class Evaluator:
    def __init__(self, run_config: RunConfig, metrics: List[MetricConfig]):
        openai.api_key = run_config.judge_api_key or os.getenv("OPENAI_KEY")
        # ensure the openai api_base points to the correct address
        # this can be updated by responders so it is necessary to set it here
        openai.api_base = DEFAULT_OPENAI_API_BASE
        if run_config.use_proxy and run_config.proxies:
            openai.proxy = {
                "http": str(run_config.proxies.http),
                "https": str(run_config.proxies.https),
            }
            _logger.debug("Proxy will be used for accessing openai API")

        self.metrics = metrics
        self.evaluator = run_config.judge
        self.evaluator_temperature = run_config.judge_temperature
        self.eval_input_tokens: List[int] = []
        self.eval_output_tokens: List[int] = []

    @Retry()
    def get_evaluation_response(self, prompt: str) -> str:
        """
        Get an evaluation from a chatbot model using a given text prompt.

        Args:
            prompt (str): The text prompt to be used for generating an evaluation.

        Returns:
            str: The generated evaluation response.

        """

        # Create a chat completion using the specified model and prompt
        completion = openai.ChatCompletion.create(
            model=self.evaluator,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.evaluator_temperature,
        )
        _logger.info(
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
        Parse a result string containing metrics and extract scores based on the criteria dictionary.

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
                    _logger.warning(str(e))
                    scores[i] = 0

        return tuple(scores)

    def get_evaluation_results(self, prompt: str) -> EvalResponse:
        eval_result = self.get_evaluation_response(prompt)

        # Parse the evaluation metrics from the result using the criteria dictionary
        metric_scores = self.parse_result(eval_result)
        # Dynamically populate the model_result_dict with scores for each criterion
        scores = []
        for i, metric in enumerate(self.metrics):
            scores.append(MetricScore(name=metric.name, score=metric_scores[i]))

        return EvalResponse(prompt=prompt, response=eval_result, scores=scores)

    def run_evaluation(self, prompts: List[EvalPrompt]) -> List[EvalResponse]:
        model_results = []
        for prompt in prompts:
            results = self.get_evaluation_results(prompt.prompt)
            model_results.append(results)

        est_cost = get_est_token_cost(
            self.evaluator, sum(self.eval_input_tokens), sum(self.eval_output_tokens)
        )
        _logger.info("Total estimated cost of evaluation: $%f", est_cost)

        return model_results
