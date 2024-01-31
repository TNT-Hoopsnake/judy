import os
import sys
import functools
from types import ModuleType
from typing import Any, List, Tuple
import openai
from openai.types.chat.chat_completion import ChatCompletion
from huggingface_hub import AsyncInferenceClient
from easyllm.prompt_utils import PROMPT_MAPPING
from easyllm.prompt_utils.base import buildBasePrompt
from easyllm.schema.base import ChatMessage

from judy.utils import Retry
from judy.config import ApiTypes, get_est_token_cost, LLMModel
from judy.config.logging import logger as log


class LLM:
    """Facade to interact with any LLM implementation."""

    def __init__(self) -> None:
        self.completion_libs = {}
        self.token_usage = {}

    def get_completion_library(self, model: LLMModel) -> ModuleType:
        """Get the completion library for the given model"""
        try:
            lib = self.completion_libs.get(model)
            if lib:
                return lib
            match model.api_type:
                case ApiTypes.OPENAI:
                    lib = openai.AsyncOpenAI(
                        api_key=model.api_key or os.getenv("OPENAI_API_KEY")
                    )
                    lib.api_base = str(model.api_base)
                    if model.use_proxy and model.proxies:
                        lib.proxy = {
                            "http": str(model.proxies.http),
                            "https": str(model.proxies.https),
                        }
                        log.debug("Proxy will be used for accessing OpenAI API")
                    lib = functools.partial(
                        lib.chat.completions.create, model=model.name
                    )
                    self.completion_libs[model] = lib
                case ApiTypes.TGI:
                    token = model.api_key or os.getenv("HUGGINGFACE_TOKEN")
                    lib = AsyncInferenceClient(model=str(model.api_base), token=token)
                    lib = lib.text_generation
                    # TODO: Handle proxy support for TGI
                    if model.use_proxy and model.proxies:
                        raise NotImplementedError(
                            "Proxy support for TGI is not yet implemented"
                        )
                    self.completion_libs[model] = lib
                case _:
                    raise ValueError(
                        f"Unable to determine completion library for api type: {model.api_base or model.name}"
                    )
            return lib
        except ValueError as e:
            log.error(str(e))
            sys.exit(1)

    @Retry()
    async def complete(
        self,
        model: LLMModel,
        messages: List[dict],
        max_new_tokens=None,
        temperature=None,
    ) -> str:
        """Complete a prompt using the given model"""
        lib = self.get_completion_library(model)
        model_response = None
        if model.api_type == ApiTypes.OPENAI:
            model_response = await lib(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
        elif model.api_type == ApiTypes.TGI:
            messages = [ChatMessage(**message) for message in messages]
            if prompt_build_func := PROMPT_MAPPING.get(model.family):
                prompt = prompt_build_func(messages)
            else:
                prompt = buildBasePrompt(messages)
            model_response = await lib(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        input_tokens, output_tokens = await self.get_token_usage(model, model_response)
        completion = await self.get_completion_text(model, model_response)
        if self.token_usage.get(model):
            self.token_usage[model]["input_tokens"] += input_tokens
            self.token_usage[model]["output_tokens"] += output_tokens
            self.token_usage[model]["total_cost"] += get_est_token_cost(
                model.name,
                model.api_type,
                input_tokens,
                output_tokens,
            )
        else:
            self.token_usage[model] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_cost": get_est_token_cost(
                    model.name,
                    model.api_type,
                    input_tokens,
                    output_tokens,
                ),
            }
        return completion

    async def get_completion_text(self, model: LLMModel, model_response: Any):
        """Get the completion text from the model response."""
        if model.api_type == ApiTypes.OPENAI:
            return model_response.choices[0].message.content
        return model_response

    async def get_token_usage(
        self, model: LLMModel, completion: str | ChatCompletion
    ) -> Tuple[int, int]:
        """Get the token usage for the given model."""
        if model.api_type == ApiTypes.OPENAI:
            return (completion.usage.prompt_tokens, completion.usage.completion_tokens)
        return (0, 0)

    def get_total_cost(self):
        """Get the total cost of the LLM usage."""
        return sum(usage["total_cost"] for usage in self.token_usage.values())
