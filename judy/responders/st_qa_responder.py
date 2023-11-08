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
    def build_model_prompts(self) -> List[STQAModelPrompt]:
        questions, answers = self.get_data_tuple()

        model_prompts = []
        for question, answer in zip(questions, answers):
            prompt = f"{SINGLE_TURN_QUESTION_PREPROMPT}{question}"
            model_prompts.append(
                STQAModelPrompt(question=question, prompt=prompt, answer=answer)
            )

        return model_prompts

    def get_model_responses(self, model_prompts):
        model_responses = []
        for model_prompt in model_prompts:
            response = self.query_model(model_prompt.prompt)
            model_responses.append(
                ModelResponse(response=response, prompt=model_prompt)
            )

        return model_responses

    def build_eval_prompts(self, model_responses: List[ModelResponse]):
        eval_prompts = []
        for model_response in model_responses:
            replacement_map = {
                "[QUESTION]": model_response.prompt.question,
                "[EXPECTED]": model_response.prompt.answer,
                "[ANSWER]": model_response.response,
            }

            eval_prompt = self.pb.build_full_prompt(ST_QA_PROMPT, replacement_map)

            eval_prompts.append(
                EvalPrompt(prompt=eval_prompt, model_response=model_response)
            )

        return eval_prompts
