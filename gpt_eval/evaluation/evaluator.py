import os
import re
from typing import List, Tuple
import openai
from gpt_eval.utils import Retry
from gpt_eval.utils.prompts import SYSTEM_PROMPT
from gpt_eval.config.settings import JudgeModels
from gpt_eval.config import (
    RunConfig,
    MetricConfig,
)
from gpt_eval.responders import (
    MetricScore,
    EvalResponse,
    EvalPrompt,
)


def get_est_token_cost(eval_model: JudgeModels, num_tokens: int) -> float:
    cost_factor = 0
    if eval_model == JudgeModels.GPT4:
        cost_factor = 0.03 / 1000.0
    elif eval_model == JudgeModels.GPT35:
        cost_factor = 0.0015 / 1000.0

    return round(num_tokens * cost_factor, 5)


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
        # logger.log(logging.INFO, f"openai request - completion tokens: {completion.usage.completion_tokens} - prompt_tokens: {completion.usage.prompt_tokens} - total: {completion.usage.total_tokens}")
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
                    min_score = min(metric.min, metric.max)
                    max_score = max(metric.min, metric.max)
                    if not min_score <= score <= max_score:
                        raise ValueError("Evaluated score was out of range")
                    scores[i] = score
                except Exception as e:
                    print(
                        f"WARNING - Evaluator provided an invalid score for '{metric.name}' ({score})"
                    )
                    print(e)
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

        return model_results
        # TODO - reimplement logging
        # total_tokens = sum(self.eval_input_tokens) + sum(self.eval_output_tokens)
        # logger.log(logging.INFO, f"- total openai usage - cost: ${get_est_token_cost(self.evaluator, total_tokens)} - tokens - combined: {total_tokens} - prompt: {sum(self.eval_input_tokens)} - completion: {sum(self.eval_output_tokens)}")
