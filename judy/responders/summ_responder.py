from typing import List
from judy.evaluation.prompts import SUMMARIZATION_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder

SUMMARY_PREPROMPT = "Provide a concise and accurate summary of the following text:\n"


class SummModelPrompt(ModelPrompt):
    context: str
    prompt: str
    answer: str


class SummarizationResponder(BaseResponder):
    async def build_model_prompt(self) -> List[SummModelPrompt]:
        docs, docs_gt = self.get_data_tuple()
        for doc, gt in zip(docs, docs_gt):
            prompt = f'{self.task_config.task_preprompt or SUMMARY_PREPROMPT}"{doc[:self._context_char_limit]}"\n'

            yield SummModelPrompt(
                prompt=prompt,
                context=doc[: self._context_char_limit],
                answer=gt,
            )

    async def get_model_response(self, model_prompt: ModelPrompt) -> ModelResponse:
        response = await self.query_chat_model(model_prompt.prompt)
        return ModelResponse(response=response, prompt=model_prompt)

    async def build_eval_prompt(self, model_response: ModelResponse):
        replacement_map = {
            "[DISCUSSION]": model_response.prompt.context,
            "[SUMMARY]": model_response.response,
        }
        eval_prompt = self.pb.build_full_prompt(
            self.task_config.eval_prompt or SUMMARIZATION_PROMPT, replacement_map
        )

        return EvalPrompt(prompt=eval_prompt, response_data=model_response)
