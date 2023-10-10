
import json
import re
import openai
import os
from gpt_eval.utils.config import (
    API_KEY,
    RESULTS_DIR,
    PROXIES,
    CRITERIA,
)

from gpt_eval.utils.prompts import SYSTEM_PROMPT


class Evaluator():
    def __init__(
            self,
            model_name,
            dataset_name, # temporarily used for storing results
            use_proxy=True,
            evaluator_model='gpt-3.5-turbo'
        ):

        if not API_KEY:
            raise RuntimeError("No OPENAI API key was found. Unable to run evaluation. Please create a .env file and add OPENAI_KEY")

        openai.api_key = API_KEY
        if use_proxy:
            openai.proxy = PROXIES

        self.use_proxy = use_proxy
        self.model_name = model_name
       
        # model evaluation responses should be saved in db table
        # linked to the original model's prompt and response
        # this is a todo, so for now continue to save the results in json
        model_results_path = self.setup_results_directory()
        self.results_path = os.path.join(model_results_path, dataset_name.split('/')[-1])
        self.evaluator = evaluator_model

        self.eval_input_tokens = []
        self.eval_output_tokens = []

    def setup_results_directory(self):
        # sanity check to ensure the top level results dir exists
        if not os.path.exists(RESULTS_DIR):
            os.mkdir(RESULTS_DIR)

        model_results_path = os.path.join(RESULTS_DIR, self.model_name)

        if not os.path.exists(model_results_path):
            os.mkdir(model_results_path)

        return model_results_path

    def get_evaluation_response(self, prompt):
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
                {
                    "role":"system",
                    "content":SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
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
        scores = [0] * len(CRITERIA)

        # Iterate through criteria and extract their scores
        for metric_name, index in CRITERIA.items():
            # Search for the metric name and its associated score
            m = re.search(f"{metric_name}: ([\d]+)", result)

            # If the metric is found, update the corresponding variable with its score
            if m is not None:
                score = m.group(1)
                scores[index] = score

        return tuple(scores)

    def get_evaluation_results(self, prompt):
        eval_result = self.get_evaluation_response(prompt)

        # Create a dictionary to store the evaluation results for each criterion
        model_result_dict = {
            'prompt':prompt,
            'eval': eval_result
        }

        # Parse the evaluation metrics from the result using the criteria dictionary
        metrics = self.parse_result(eval_result)
        # Dynamically populate the model_result_dict with scores for each criterion
        for criterion, index in CRITERIA.items():
            model_result_dict[criterion] = metrics[index]

        return model_result_dict
    
    def run_evaluation(self, prompts):
        model_results = []
        for prompt in prompts:
            results = self.get_evaluation_results(prompt)
            model_results.append(results)

        with open(os.path.join(RESULTS_DIR, f'{self.results_path}-results.json'), 'w+') as fn:
            json.dump(model_results, fn, indent=4)

        total_tokens = sum(self.eval_input_tokens) + sum(self.eval_output_tokens)
        # logger.log(logging.INFO, f"- total openai usage - cost: ${get_est_token_cost(self.evaluator, total_tokens)} - tokens - combined: {total_tokens} - prompt: {sum(self.eval_input_tokens)} - completion: {sum(self.eval_output_tokens)}")




