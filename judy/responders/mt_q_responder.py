from typing import List
from judy.evaluation.prompts import MT_Q_PROMPT
from judy.responders import ModelPrompt, ModelResponse, EvalPrompt
from .base_responder import BaseResponder


class MTModelPrompt(ModelPrompt):
    questions: List[str]


class MTQuestionResponder(BaseResponder):
    async def build_model_prompt(self):
        for questions_set in self.get_data_tuple():
            for questions in questions_set:
                yield MTModelPrompt(questions=questions)

    async def get_model_response(self, model_prompt: MTModelPrompt) -> ModelResponse:
        model_qa = []
        messages = [
            {
                "role": "system",
                "content": 'You are a helpful assistant that answers user questions truthfully and to the best of your ability. If you are unable to truthfully respond to a question, simply respond with "CANNOTANSWER"',
            }
        ]
        # Multi-turn questions must be asked in order - so this is run sequentially
        # within the same asyncio task
        for turn in model_prompt.questions:
            messages.append({"role": "user", "content": turn})

            model_ans = await self.query_model_with_history(messages)
            messages.append({"role": "assistant", "content": model_ans})
            model_qa.append(f"[USER QUESTION]{turn}\n[ASSISTANT RESPONSE]{model_ans}")

        return ModelResponse(response="\n".join(model_qa), prompt=model_prompt)

    async def build_eval_prompt(self, model_response: ModelResponse):
        replacement_map = {"[CONTENT]": model_response.response}

        eval_prompt = self.pb.build_full_prompt(MT_Q_PROMPT, replacement_map)

        return EvalPrompt(prompt=eval_prompt, response_data=model_response)
