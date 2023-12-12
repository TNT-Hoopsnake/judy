from typing import List
from judy.utils.prompts import ST_Q_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder

SINGLE_TURN_QUESTION_PREPROMPT = "Answer the following question:\n"


class STModelPrompt(ModelPrompt):
    prompt: str
    question: str


class STQuestionResponder(BaseResponder):
    async def build_model_prompt(self) -> List[STModelPrompt]:
        questions = self.get_data_tuple()[0]
        for question in questions:
            prompt = f"{SINGLE_TURN_QUESTION_PREPROMPT}{question}"

            yield STModelPrompt(
                question=question,
                prompt=prompt,
            )

    async def get_model_response(self, model_prompt: STModelPrompt) -> ModelResponse:
        response = await self.query_model(model_prompt.prompt)
        return ModelResponse(response=response, prompt=model_prompt)

    async def build_eval_prompt(self, model_response: ModelResponse) -> EvalPrompt:
        replacement_map = {
            "[QUESTION]": model_response.prompt.question,
            "[ANSWER]": model_response.response,
        }

        eval_prompt = self.pb.build_full_prompt(ST_Q_PROMPT, replacement_map)

        return EvalPrompt(prompt=eval_prompt, response_data=model_response)
