from typing import List
from gpt_eval.utils.prompts import MT_QAC_PROMPT
from gpt_eval.config.config_models import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder


class MTQACModelPrompt(ModelPrompt):
    questions: List[str]
    context: str
    gt_answers: List[List[str]]


class MTQACModelResponse(ModelResponse):
    gt_response: str


class MTQuestionAnswerContextResponder(BaseResponder):
    def build_model_prompts(self) -> List[MTQACModelPrompt]:
        questions_list, answers_list, contexts = self.data

        model_prompts = []
        for questions, answers, context in zip(questions_list, answers_list, contexts):
            context = context[: self._context_char_limit]
            model_prompts.append(
                MTQACModelPrompt(
                    questions=questions, context=context, gt_answers=answers
                )
            )

        return model_prompts

    def get_model_responses(
        self, model_prompts: List[MTQACModelPrompt]
    ) -> List[MTQACModelResponse]:
        model_responses = []
        for model_prompt in model_prompts:
            model_qa = []
            expected_qa = []
            messages = [
                {
                    "role": "system",
                    "content": 'You are a helpful assistant that answers user questions truthfully and to the best of your ability. If you are unable to truthfully answer a question, simply respond with "CANNOTANSWER"',
                },
                {
                    "role": "user",
                    "content": f"Study and understand this piece of text. I will follow up with questions about it. {model_prompt.context}",
                },
                {
                    "role": "assistant",
                    "content": "Understood. Please ask me questions about the given piece of text",
                },
            ]
            for question, expected_answer in zip(
                model_prompt.questions, model_prompt.gt_answers
            ):
                messages.append({"role": "user", "content": question})

                model_ans = self.query_chat_model(messages)
                messages.append({"role": "assistant", "content": model_ans})
                model_qa.append(
                    f"[USER QUESTION]{question}\n[ASSISTANT RESPONSE]{model_ans}"
                )
                expected_qa.append(
                    f"[USER QUESTION]{question}\n[EXPECTED ANSWER]{expected_answer}"
                )

            model_responses.append(
                MTQACModelResponse(
                    response="\n".join(model_qa),
                    gt_response="\n".join(expected_qa),
                    prompt=model_prompt,
                )
            )

        return model_responses

    def build_eval_prompts(
        self, model_responses: List[MTQACModelResponse]
    ) -> List[EvalPrompt]:
        eval_prompts = []
        for model_response in model_responses:
            replacement_map = {
                "[ANSWER]": model_response.gt_response,
                "[CONTEXT]": model_response.prompt.context,
                "[CONTENT]": model_response.response,
            }
            eval_prompt = self.pb.build_full_prompt(MT_QAC_PROMPT, replacement_map)

            eval_prompts.append(
                EvalPrompt(prompt=eval_prompt, model_response=model_response)
            )

        return eval_prompts
