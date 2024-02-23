from typing import List
from judy.evaluation.prompts import ST_QAC_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder

SINGLE_TURN_QUESTION_ANSWER_CONTEXT_PREPROMPT = (
    "Using the context above, answer the following question:\n"
)


class STQACModelPrompt(ModelPrompt):
    question: str
    prompt: str
    context: str
    answer: str


class STQuestionAnswerContextResponder(BaseResponder):
    async def build_model_prompt(self) -> List[STQACModelPrompt]:
        questions, answers, contexts = self.get_data_tuple()

        for question, answer, context in zip(questions, answers, contexts):
            context = context[: self._context_char_limit]
            append_char = ""
            if not question.endswith("?"):
                append_char = "?"
            prompt = f"[CONTEXT]: {context}\n{self.task_config.task_preprompt or SINGLE_TURN_QUESTION_ANSWER_CONTEXT_PREPROMPT}{question}{append_char}"
            yield STQACModelPrompt(
                question=question,
                prompt=prompt,
                context=context,
                answer=answer,
            )

    async def get_model_response(self, model_prompt: STQACModelPrompt) -> ModelResponse:
        response = await self.query_chat_model(model_prompt.prompt)
        return ModelResponse(response=response, prompt=model_prompt)

    async def build_eval_prompt(self, model_response: ModelResponse) -> EvalPrompt:
        replacement_map = {
            "[QUESTION]": model_response.prompt.question,
            "[ANSWER]": model_response.response,
            "[CONTEXT]": model_response.prompt.context,
        }

        eval_prompt = self.pb.build_full_prompt(
            self.task_config.eval_prompt or ST_QAC_PROMPT, replacement_map
        )

        return EvalPrompt(prompt=eval_prompt, response_data=model_response)
