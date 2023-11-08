from typing import List
from judy.utils.prompts import MT_Q_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder


class MTModelPrompt(ModelPrompt):
    questions: List[str]


class MTQuestionResponder(BaseResponder):
    def build_model_prompts(self):
        prompts = []
        for questions_set in self.get_data_tuple():
            for questions in questions_set:
                prompts.append(MTModelPrompt(questions=questions))

        return prompts

    def get_model_responses(
        self, model_prompts: List[MTModelPrompt]
    ) -> List[ModelResponse]:
        model_responses = []
        for turns in model_prompts:
            model_qa = []
            messages = [
                {
                    "role": "system",
                    "content": 'You are a helpful assistant that answers user questions truthfully and to the best of your ability. If you are unable to truthfully respond to a question, simply respond with "CANNOTANSWER"',
                }
            ]
            for turn in turns.questions:
                messages.append({"role": "user", "content": turn})

                model_ans = self.query_chat_model(messages)
                messages.append({"role": "assistant", "content": model_ans})
                model_qa.append(
                    f"[USER QUESTION]{turn}\n[ASSISTANT RESPONSE]{model_ans}"
                )

            model_responses.append(
                ModelResponse(response="\n".join(model_qa), prompt=turns)
            )

        return model_responses

    def build_eval_prompts(self, model_responses: List[ModelResponse]):
        eval_prompts = []
        for model_response in model_responses:
            replacement_map = {"[CONTENT]": model_response.response}

            eval_prompt = self.pb.build_full_prompt(MT_Q_PROMPT, replacement_map)

            eval_prompts.append(
                EvalPrompt(prompt=eval_prompt, model_response=model_response)
            )

        return eval_prompts
