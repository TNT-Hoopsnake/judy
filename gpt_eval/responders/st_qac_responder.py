from typing import List
from gpt_eval.utils.prompts import ST_QAC_PROMPT
from gpt_eval.config.config_models import ModelPrompt, ModelResponse, EvalPrompt
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
    def build_model_prompts(self) -> List[STQACModelPrompt]:
        questions, answers, contexts = self.data

        model_prompts = []
        for question, answer, context in zip(questions, answers, contexts):
            context = context[: self._context_char_limit]
            append_char = ""
            if not question.endswith("?"):
                append_char = "?"
            prompt = f"[CONTEXT]: {context}\n{SINGLE_TURN_QUESTION_ANSWER_CONTEXT_PREPROMPT}{question}{append_char}"
            model_prompts.append(
                STQACModelPrompt(
                    question=question,
                    prompt=prompt,
                    context=context,
                    answer=answer,
                )
            )

        return model_prompts

    def get_model_responses(
        self, model_prompts: List[STQACModelPrompt]
    ) -> List[ModelResponse]:
        model_responses = []
        for model_prompt in model_prompts:
            response = self.query_model(model_prompt.prompt)
            # eventually should be linked via sql tables
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
                "[CONTEXT]": model_response.prompt.context,
            }

            eval_prompt = self.pb.build_full_prompt(ST_QAC_PROMPT, replacement_map)

            eval_prompts.append(
                EvalPrompt(prompt=eval_prompt, model_response=model_response)
            )

        return eval_prompts
