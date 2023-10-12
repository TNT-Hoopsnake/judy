from abc import ABC, abstractmethod
from easyllm.clients import huggingface
import openai
from gpt_eval.config import ApiTypes

def get_completion_library(api_type, api_base):
    if api_type == ApiTypes.OPENAI:
        lib = openai
    elif api_type == ApiTypes.TGI:
        lib= huggingface
    else:
        raise ValueError(f'Unable to determine completion library for api type: {api_type}')
    
    lib.api_base = api_base
    return lib


class BaseResponder(ABC):
    def __init__(
            self, 
            data,
            api_type,
            api_base,
            temperature,
            max_tokens,
            context_char_limit
        ):
        # should eventually read these values from a config file
        self.data = data
        self._api_type = api_type
        self._api_base = api_base
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._context_char_limit = context_char_limit

    def query_model(self, prompt):
        lib = get_completion_library(self._api_type, self._api_base)

        completion = lib.Completion.create(
            prompt=prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        output = completion['choices'][0]['text']

        return output

    def query_chat_model(self, chat_history):
        lib = get_completion_library(self._api_type, self._api_base)

        completion = lib.ChatCompletion.create(
            messages=chat_history,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        output = completion['choices'][0]['message']['content']

        return output 

    def get_evaluation_prompts(self):
        prompts = self.build_model_prompts()
        responses =  self.get_model_responses(prompts)
        eval_prompts = self.build_eval_prompts(responses)
        return eval_prompts


    @abstractmethod
    def build_model_prompts(self):
        pass
    
    @abstractmethod
    def get_model_responses(self, prompts):
        pass

    @abstractmethod
    def build_eval_prompts(self, model_response, original):
        pass

    


    




