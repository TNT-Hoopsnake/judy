from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from types import ModuleType
import sys

import openai
from easyllm.clients import huggingface
from easyllm.prompt_utils import PROMPT_MAPPING
from easyllm.prompt_utils.base import buildBasePrompt
from easyllm.schema.base import ChatMessage

from judy.config import ApiTypes
from judy.utils import PromptBuilder
from judy.config import EvaluatedModel
from judy.dataset import BaseFormattedData
from judy.responders import (
    ModelPrompt,
    ModelResponse,
    EvalPrompt,
)
from judy.utils import Retry
from judy.config.logging import logger as log


class BaseResponder(ABC):
    def __init__(
        self,
        data: BaseFormattedData,
        prompt_builder: PromptBuilder,
        model_config: EvaluatedModel,
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
        chat_history = [{"role": "user", "content": prompt}]

        return self.query_chat_model(chat_history)

    def get_data_tuple(self) -> Tuple[Any]:
        return tuple(self.data.model_dump().values())

    @Retry()
    def query_chat_model(self, chat_history: List[dict]):
        lib = self.get_completion_library()

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
        output = completion["choices"][0]["text"]

        return output

    def get_evaluation_prompts(self):
        prompts = self.build_model_prompts()
        responses = self.get_model_responses(prompts)
        eval_prompts = self.build_eval_prompts(responses)
        return eval_prompts

    def get_completion_library(self) -> ModuleType:
        try:
            match self._api_type:
                case ApiTypes.OPENAI:
                    lib = openai
                    # openai lib requires api_key to be set, even if we're not accessing the actual OAI api
                    openai.api_key = ""
                case ApiTypes.TGI:
                    lib = huggingface
                    # used to suppress annoying warning from easyllm
                    lib.prompt_builder = lambda x: x
                case _:
                    raise ValueError(
                        f"Unable to determine completion library for api type: {self._api_type}"
                    )

            lib.api_base = self._api_base
            return lib
        except ValueError as e:
            log.error(str(e))
            sys.exit(1)

    @abstractmethod
    def build_model_prompts(self) -> List[ModelPrompt]:
        pass

    @abstractmethod
    def get_model_responses(
        self, model_prompts: List[ModelPrompt]
    ) -> List[ModelResponse]:
        pass

    @abstractmethod
    def build_eval_prompts(
        self, model_responses: List[ModelResponse]
    ) -> List[EvalPrompt]:
        pass
