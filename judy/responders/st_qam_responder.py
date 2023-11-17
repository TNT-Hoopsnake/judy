from typing import List
from judy.utils.prompts import ST_QA_PROMPT
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
    def build_model_prompts(self) -> List[STQAMModelPrompt]:
        questions, answers, metrics = self.get_data_tuple()

        model_prompts = []
        for question, answer, metric_list in zip(questions, answers, metrics):
            prompt = question
            model_prompts.append(
                STQAMModelPrompt(
                    question=question, prompt=prompt, answer=answer, metrics=metric_list
                )
            )

        return model_prompts

    def build_eval_prompts(self, model_responses: List[ModelResponse]):
        eval_prompts = []
        for model_response in model_responses:
            replacement_map = {
                "[QUESTION]": model_response.prompt.question,
                "[EXPECTED]": model_response.prompt.answer,
                "[ANSWER]": model_response.response,
            }

            eval_prompt = self.pb.build_full_prompt(
                ST_QA_PROMPT, replacement_map, model_response.prompt.metrics
            )

            eval_prompts.append(
                EvalPrompt(prompt=eval_prompt, response_data=model_response)
            )

        return eval_prompts
