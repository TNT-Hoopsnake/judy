from typing import List
from judy.evaluation.prompts import ST_QA_PROMPT
from judy.responders import (
    STQuestionAnswerResponder,
    ModelResponse,
    EvalPrompt,
    ModelPrompt,
)


class STQAMModelPrompt(ModelPrompt):
    question: str
    prompt: str
    answer: str
    metrics: List[str]


class STQuestionAnswerMetricResponder(STQuestionAnswerResponder):
    async def build_model_prompt(self) -> List[STQAMModelPrompt]:
        questions, answers, metrics = self.get_data_tuple()

        for question, answer, metric_list in zip(questions, answers, metrics):
            prompt = question
            yield STQAMModelPrompt(
                question=question, prompt=prompt, answer=answer, metrics=metric_list
            )

    async def build_eval_prompt(self, model_response: ModelResponse):
        replacement_map = {
            "[QUESTION]": model_response.prompt.question,
            "[EXPECTED]": model_response.prompt.answer,
            "[ANSWER]": model_response.response,
        }

        eval_prompt = self.pb.build_full_prompt(
            ST_QA_PROMPT, replacement_map, model_response.prompt.metrics
        )

        return EvalPrompt(prompt=eval_prompt, response_data=model_response)
