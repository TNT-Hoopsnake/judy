
import json
import re
import openai
import os
from gpt_eval.utils.prompts import SYSTEM_PROMPT

def get_est_token_cost(eval_model, num_tokens):
    if eval_model == 'gpt-4':
        cost_factor = (0.03 / 1000) 
    elif eval_model == 'gpt-3.5-turbo':
        cost_factor = (0.0015 / 1000)

    return round(num_tokens * cost_factor, 5)



class Evaluator:
    def __init__(
            self,
            system_config,
            metrics
        ):

        openai.api_key = system_config.judge_api_key or os.getenv('OPENAI_KEY')
        if system_config.use_proxy:
            openai.proxy = system_config.proxies

        self.metrics = metrics
        self.evaluator = system_config.judge
        self.evaluator_temperature = system_config.judge_temperature
        self.eval_input_tokens = []
        self.eval_output_tokens = []


    def get_evaluation_response(self, prompt):
        """
        Get an evaluation from a chatbot model using a given text prompt.

        Args:
            prompt (str): The text prompt to be used for generating an evaluation.

        Returns:
            str: The generated evaluation response.

        """
        # TODO - add error handling to this request

        # Create a chat completion using the specified model and prompt
        completion = openai.ChatCompletion.create(
            model=self.evaluator,
            messages=[
                {
                    "role":"system",
                    "content":SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=self.evaluator_temperature
        )
        # logger.log(logging.INFO, f"openai request - completion tokens: {completion.usage.completion_tokens} - prompt_tokens: {completion.usage.prompt_tokens} - total: {completion.usage.total_tokens}")
        self.eval_input_tokens.append(completion.usage.prompt_tokens)
        self.eval_output_tokens.append(completion.usage.completion_tokens)
        # Extract and return the content of the generated evaluation response
        return completion.choices[0].message.content

    def parse_result(self, result):
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
            m = re.search(f"{metric.name}: ([\d]+)", result)

            # If the metric is found, update the corresponding variable with its score
            if m is not None:
                score = m.group(1)
                scores[i] = score

        return tuple(scores)

    def get_evaluation_results(self, prompt):
        eval_result = self.get_evaluation_response(prompt)

        # Create a dictionary to store the evaluation results for each criterion
        model_result_dict = {
            'prompt':prompt,
            'eval': eval_result
        }

        # Parse the evaluation metrics from the result using the criteria dictionary
        metric_scores = self.parse_result(eval_result)
        # Dynamically populate the model_result_dict with scores for each criterion
        for i, metric in enumerate(self.metrics):
            model_result_dict[metric.name] = metric_scores[i]

        return model_result_dict
    
    def run_evaluation(self, prompts):
        model_results = []
        for prompt in prompts:
            results = self.get_evaluation_results(prompt['eval_prompt'])
            model_results.append(results)


        return model_results
        # TODO - reimplement logging
        # total_tokens = sum(self.eval_input_tokens) + sum(self.eval_output_tokens)
        # logger.log(logging.INFO, f"- total openai usage - cost: ${get_est_token_cost(self.evaluator, total_tokens)} - tokens - combined: {total_tokens} - prompt: {sum(self.eval_input_tokens)} - completion: {sum(self.eval_output_tokens)}")




