from abc import ABC, abstractmethod
from typing import Tuple, Any, List

from judy.evaluation import PromptBuilder
from judy.config import EvaluatedModel, TaskConfig
from judy.dataset import BaseFormattedData
from judy.responders import (
    ModelPrompt,
    ModelResponse,
    EvalPrompt,
)
from judy.utils import LLM, LLMModel


class BaseResponder(ABC):
    def __init__(
        self,
        llm: LLM,
        data: BaseFormattedData,
        prompt_builder: PromptBuilder,
        model_config: EvaluatedModel,
        task_config: TaskConfig,
    ):
        self.data = data
        self.pb = prompt_builder
        self.model_id = model_config.id
        # config file values
        self._model_family = model_config.family
        self._api_type = model_config.api_type
        self._temperature = model_config.temperature
        self._max_tokens = model_config.max_tokens
        self._context_char_limit = model_config.context_char_limit
        self.llm = llm
        self.llm_model = LLMModel(
            api_type=self._api_type,
            api_base=model_config.api_base,
            api_key=model_config.api_key,
            name=self.model_id,
            family=self._model_family,
        )
        self.task_config = task_config

    async def query_model_with_history(self, history: List[dict]):
        return await self.llm.complete(
            self.llm_model,
            messages=history,
            max_new_tokens=self._max_tokens,
            temperature=self._temperature,
        )

    async def query_chat_model(self, prompt):
        """Query the model with the given prompt."""
        chat_history = [{"role": "user", "content": prompt}]
        return await self.query_model_with_history(chat_history)

    def get_data_tuple(self) -> Tuple[Any]:
        """Get the data tuple from the data model."""
        return tuple(self.data.model_dump().values())

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

    @abstractmethod
    async def build_model_prompt(self) -> ModelPrompt:
        """Build the model prompt from the data tuple."""

    @abstractmethod
    async def get_model_response(self, model_prompt: ModelPrompt) -> ModelResponse:
        """Prompt the model with a single prompt and get the model response"""

    @abstractmethod
    async def build_eval_prompt(self, model_response: ModelResponse) -> EvalPrompt:
        """Build the evaluation prompt from the model response."""
