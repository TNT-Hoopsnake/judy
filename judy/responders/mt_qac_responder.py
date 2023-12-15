from typing import List
from judy.evaluation.prompts import MT_QAC_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder


class MTQACModelPrompt(ModelPrompt):
    questions: List[str]
    context: str
    gt_answers: List[List[str]]


class MTQACModelResponse(ModelResponse):
    gt_response: str


class MTQuestionAnswerContextResponder(BaseResponder):
    async def build_model_prompt(self) -> List[MTQACModelPrompt]:
        questions_list, contexts, answers_list = self.get_data_tuple()
        for questions, answers, context in zip(questions_list, answers_list, contexts):
            context = context[: self._context_char_limit]

            yield MTQACModelPrompt(
                questions=questions, context=context, gt_answers=answers
            )

    async def get_model_response(
        self, model_prompt: MTQACModelPrompt
    ) -> MTQACModelResponse:
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

            model_ans = await self.query_model_with_history(messages)
            messages.append({"role": "assistant", "content": model_ans})
            model_qa.append(
                f"[USER QUESTION]{question}\n[ASSISTANT RESPONSE]{model_ans}"
            )
            expected_qa.append(
                f"[USER QUESTION]{question}\n[EXPECTED ANSWER]{expected_answer}"
            )

        return MTQACModelResponse(
            response="\n".join(model_qa),
            gt_response="\n".join(expected_qa),
            prompt=model_prompt,
        )

    async def build_eval_prompt(self, model_response: MTQACModelResponse) -> EvalPrompt:
        replacement_map = {
            "[ANSWER]": model_response.gt_response,
            "[CONTEXT]": model_response.prompt.context,
            "[CONTENT]": model_response.response,
        }
        eval_prompt = self.pb.build_full_prompt(MT_QAC_PROMPT, replacement_map)

        return EvalPrompt(prompt=eval_prompt, response_data=model_response)
