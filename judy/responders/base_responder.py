from abc import ABC, abstractmethod
from typing import List, Tuple, Any
from types import ModuleType
import sys

import openai
from easyllm.prompt_utils import PROMPT_MAPPING
from easyllm.prompt_utils.base import buildBasePrompt
from easyllm.schema.base import ChatMessage
from huggingface_hub import AsyncInferenceClient

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
        self.model_id = model_config.id
        # config file values
        self._model_family = model_config.family
        self._api_type = model_config.api_type
        self._api_base = str(model_config.api_base)
        self._temperature = model_config.temperature
        self._max_tokens = model_config.max_tokens
        self._context_char_limit = model_config.context_char_limit
        self.completion_libs = {}

    async def query_model(self, prompt):
        """Query the model with the given prompt."""
        chat_history = [{"role": "user", "content": prompt}]

        return await self.query_chat_model(chat_history)

    def get_data_tuple(self) -> Tuple[Any]:
        """Get the data tuple from the data model."""
        return tuple(self.data.model_dump().values())

    @Retry()
    async def query_chat_model(self, chat_history: List[dict]):
        """Query the chat model with the given chat history."""
        lib = self.get_completion_library()

        messages = [ChatMessage(**message) for message in chat_history]

        if prompt_build_func := PROMPT_MAPPING.get(self._model_family):
            prompt = prompt_build_func(messages)
        else:
            prompt = buildBasePrompt(messages)

        # easyllm will append the model name to the base url when using TGI completion lib
        # this isnt wanted behaviour, so we set model name to None in those cases
        model_name = None
        complete_func = lib.text_generation
        if self._api_type == ApiTypes.OPENAI:
            # openai lib requires a model name when using Completion
            # we only set it when that lib is in use
            model_name = self._model_family
            complete_func = lib.Completion.create
        output = await complete_func(
            model=model_name,
            prompt=prompt,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        # output = completion["choices"][0]["text"]

        return output

    async def get_model_prompts(self):
        """Get model prompts for the given data tuple."""
        prompts = []
        async for prompt in self.build_model_prompt():
            prompts.append(prompt)
        return prompts

    async def get_responses_for_prompts(self, prompts, progress_bar):
        """Get model responses for a given list of prompts"""
        responses = []
        for prompt in prompts:
            responses.append(await self.get_model_response(prompt))
        progress_bar.update()
        return responses

    def get_completion_library(self) -> ModuleType:
        """Get the completion library for the given api type."""
        try:
            match self._api_type:
                case ApiTypes.OPENAI:
                    if self.completion_libs.get(ApiTypes.OPENAI):
                        lib = self.completion_libs[ApiTypes.OPENAI]
                    else:
                        lib = openai.AsyncOpenAI
                        # openai lib requires api_key to be set, even if we're not accessing the actual OAI api
                        openai.api_key = ""
                        lib.api_base = self._api_base
                        self.completion_libs[ApiTypes.OPENAI] = lib
                case ApiTypes.TGI:
                    if self.completion_libs.get(ApiTypes.TGI):
                        lib = self.completion_libs[ApiTypes.TGI]
                    else:
                        lib = AsyncInferenceClient(model=self._api_base)
                        # used to suppress annoying warning from easyllm
                        # lib.prompt_builder = lambda x: x
                        self.completion_libs[ApiTypes.TGI] = lib
                case _:
                    raise ValueError(
                        f"Unable to determine completion library for api type: {self._api_type}"
                    )

            # lib.api_base = self._api_base
            return lib
        except ValueError as e:
            log.error(str(e))
            sys.exit(1)

    @abstractmethod
    async def build_model_prompt(self) -> ModelPrompt:
        """Build the model prompt from the data tuple."""

    @abstractmethod
    async def get_model_response(self, model_prompt: ModelPrompt) -> ModelResponse:
        """Prompt the model with a single prompt and get the model response"""

    @abstractmethod
    async def build_eval_prompt(self, model_response: ModelResponse) -> EvalPrompt:
        """Build the evaluation prompt from the model response."""
