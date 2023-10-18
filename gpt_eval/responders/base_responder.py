from abc import ABC, abstractmethod
from gpt_eval.utils import get_completion_library, Retry
from easyllm.prompt_utils.base import buildBasePrompt
from easyllm.prompt_utils import PROMPT_MAPPING
from easyllm.schema.base import ChatMessage
from gpt_eval.config import ApiTypes

class BaseResponder(ABC):
    def __init__(
            self, 
            data,
            prompt_builder,
            model_config
        ):
        self.data = data
        self.pb = prompt_builder
        # config file values
        self._model_family = model_config.family
        self._api_type = model_config.api_type
        self._api_base = str(model_config.api_base)
        self._temperature = model_config.temperature
        self._max_tokens = model_config.max_tokens
        self._context_char_limit = model_config.context_char_limit

    def query_model(self, prompt):
        chat_history = [
            {
                'role':'user',
                'content': prompt
            }
        ]

        return self.query_chat_model(chat_history)

    @Retry()
    def query_chat_model(self, chat_history):
        lib = get_completion_library(self._api_type, self._api_base)

        messages = [ChatMessage(**message) for message in chat_history]

        if prompt_build_func := PROMPT_MAPPING.get(self._model_family):
            prompt = prompt_build_func(messages)
        else:
            prompt = buildBasePrompt(messages)

        # easyllm will append the model name to the base url when using TGI completion lib
        # this isnt wanted behaviour, so we set model name to None in those cases
        model_name = None
        if self._api_type == ApiTypes.OPENAI:
            # openai lib requires a model name when using Completion
            # we only set it when that lib is in use
            model_name = self._model_family

        completion = lib.Completion.create(
            model=model_name,
            prompt=prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        output = completion['choices'][0]['text']

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

    


    




