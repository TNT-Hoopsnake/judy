from typing import List
from judy.utils.prompts import SUMMARIZATION_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder

SUMMARY_PREPROMPT = "Provide a concise and accurate summary of the following text:\n"


class SummModelPrompt(ModelPrompt):
    context: str
    prompt: str
    answer: str


class SummarizationResponder(BaseResponder):
    def build_model_prompts(self) -> List[SummModelPrompt]:
        docs, docs_gt = self.get_data_tuple()
        model_prompts = []
        for doc, gt in zip(docs, docs_gt):
            prompt = f'{SUMMARY_PREPROMPT}"{doc[:self._context_char_limit]}"\n'

            model_prompts.append(
                SummModelPrompt(
                    prompt=prompt,
                    context=doc[: self._context_char_limit],
                    answer=gt,
                )
            )

        return model_prompts

    def get_model_responses(
        self, model_prompts: List[ModelPrompt]
    ) -> List[ModelResponse]:
        model_responses = []
        for model_prompt in model_prompts:
            response = self.query_model(model_prompt.prompt)
            # eventually should be linked via sql tables
            model_responses.append(
                ModelResponse(response=response, prompt=model_prompt)
            )

        return model_responses

    def build_eval_prompts(self, model_responses: List[ModelResponse]):
        eval_prompts = []
        for model_response in model_responses:
            replacement_map = {
                "[DISCUSSION]": model_response.prompt.context,
                "[SUMMARY]": model_response.response,
            }
            eval_prompt = self.pb.build_full_prompt(
                SUMMARIZATION_PROMPT, replacement_map
            )

            eval_prompts.append(
                EvalPrompt(prompt=eval_prompt, response_data=model_response)
            )

        return eval_prompts
