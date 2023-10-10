from abc import ABC, abstractmethod
from easyllm.clients import huggingface
from gpt_eval.data.loader import load_formatted_data
import openai

MODEL_MAX_TOKENS = 600
TEMPERATURE = 0

class BaseResponder(ABC):
    def __init__(self, model, api_base, ds_name, ds_vers=None, api_key=None):
        # should eventually read these values from a config file
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.ds_name = ds_name
        self.ds_vers = ds_vers    

    def query_model(self, prompt):
        # will swap between openai vs huggingface api structure using value in config file
        # values (eventually) from config file
        # openai.api_key=self.api_key # should be read from env var
        # openai.api_base=self.api_base
        # huggingface.api_base=self.api_base


        # eventually replace with ChatCompletion but just use Completion for now
        completion = huggingface.Completion.create(
            # including model here adds it to the base url (NOT WHAT WE WANT!)
            prompt=prompt,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        output = completion['choices'][0]['text']

        return output

    def query_chat_model(self, chat_history):
        # openai.api_key=API_KEY # should be read from env var
        # openai.api_base=self.api_base
        # huggingface.api_base=self.api_base

        completion = huggingface.ChatCompletion.create(
            messages=chat_history,
            max_tokens=MODEL_MAX_TOKENS,
            temperature=TEMPERATURE,
        )
        output = completion['choices'][0]['message']['content']

        return output 

    def get_evaluation_prompts(self, num_prompts):
        # todo - use docs_gt as pseudo model response

        # formatted_data will be a tuple containing different values depending on the dataset used
        formatted_data = load_formatted_data(num_prompts, self.ds_name, self.ds_vers)

        # build_model_prompts is implemented per responder class and handles the different tuple sizes output by load_formatted_data
        prompts = self.build_model_prompts(formatted_data)
        responses =  self.get_model_responses(prompts)
        eval_prompts = self.build_eval_prompts(responses)
        return eval_prompts


    @abstractmethod
    def build_model_prompts(self, formatted_data):
        pass
    
    @abstractmethod
    def get_model_responses(self, prompts):
        pass

    @abstractmethod
    def build_eval_prompts(self, model_response, original):
        pass

    


    




