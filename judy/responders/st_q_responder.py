from typing import List
from judy.utils.prompts import ST_Q_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder

SINGLE_TURN_QUESTION_PREPROMPT = "Answer the following question:\n"


class STModelPrompt(ModelPrompt):
    prompt: str
    question: str


class STQuestionResponder(BaseResponder):
    def build_model_prompts(self) -> List[STModelPrompt]:
        questions = self.get_data_tuple()[0]

        model_prompts = []
        for question in questions:
            prompt = f"{SINGLE_TURN_QUESTION_PREPROMPT}{question}"
            model_prompts.append(
                STModelPrompt(
                    question=question,
                    prompt=prompt,
                )
            )

        return model_prompts

    def get_model_responses(
        self, model_prompts: List[STModelPrompt]
    ) -> List[ModelResponse]:
        model_responses = []
        for model_prompt in model_prompts:
            response = self.query_model(model_prompt.prompt)
            model_responses.append(
                ModelResponse(response=response, prompt=model_prompt)
            )

        return model_responses

    def build_eval_prompts(
        self, model_responses: List[ModelResponse]
    ) -> List[EvalPrompt]:
        eval_prompts = []
        for model_response in model_responses:
            replacement_map = {
                "[QUESTION]": model_response.prompt.question,
                "[ANSWER]": model_response.response,
            }

            eval_prompt = self.pb.build_full_prompt(ST_Q_PROMPT, replacement_map)

            eval_prompts.append(
                EvalPrompt(prompt=eval_prompt, response_data=model_response)
            )

        return eval_prompts
