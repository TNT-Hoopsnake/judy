from typing import List
from judy.utils.prompts import ST_QA_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder

SINGLE_TURN_QUESTION_PREPROMPT = "Answer the following question:\n"


class STQAModelPrompt(ModelPrompt):
    question: str
    prompt: str
    answer: str


class STQuestionAnswerResponder(BaseResponder):
    async def build_model_prompt(self) -> List[STQAModelPrompt]:
        questions, answers = self.get_data_tuple()

        for question, answer in zip(questions, answers):
            prompt = f"{SINGLE_TURN_QUESTION_PREPROMPT}{question}"
            yield STQAModelPrompt(question=question, prompt=prompt, answer=answer)

    async def get_model_response(self, model_prompt: STQAModelPrompt):
        response = await self.query_model(model_prompt.prompt)
        return ModelResponse(response=response, prompt=model_prompt)

    async def build_eval_prompt(self, model_response: ModelResponse):
        replacement_map = {
            "[QUESTION]": model_response.prompt.question,
            "[EXPECTED]": model_response.prompt.answer,
            "[ANSWER]": model_response.response,
        }

        eval_prompt = self.pb.build_full_prompt(ST_QA_PROMPT, replacement_map)

        return EvalPrompt(prompt=eval_prompt, response_data=model_response)
