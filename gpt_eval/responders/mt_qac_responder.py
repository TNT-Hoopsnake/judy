from gpt_eval.utils.prompts import MT_QAC_PROMPT
from .base_responder import BaseResponder


class MTQuestionAnswerContextResponder(BaseResponder):
    def build_model_prompts(self):
        questions_list, answers_list, contexts = self.data

        model_prompts = []
        for questions, answers, context in zip(questions_list, answers_list, contexts):
            context = context[: self._context_char_limit]
            model_prompts.append(
                {"questions": questions, "context": context, "gt_answers": answers}
            )

        return model_prompts

    def get_model_responses(self, prompt_contexts):
        model_responses = []
        for prompt_context in prompt_contexts:
            model_qa = []
            expected_qa = []
            messages = [
                {
                    "role": "system",
                    "content": 'You are a helpful assistant that answers user questions truthfully and to the best of your ability. If you are unable to truthfully answer a question, simply respond with "CANNOTANSWER"',
                },
                {
                    "role": "user",
                    "content": f"Study and understand this piece of text. I will follow up with questions about it. {prompt_context['context']}",
                },
                {
                    "role": "assistant",
                    "content": "Understood. Please ask me questions about the given piece of text",
                },
            ]
            for question, expected_answer in zip(
                prompt_context["questions"], prompt_context["gt_answers"]
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
                {
                    "model_responses": "\n".join(model_qa),
                    "expected_answers": "\n".join(expected_qa),
                    **prompt_context,
                }
            )

        return model_responses

    def build_eval_prompts(self, prompt_context_responses):
        eval_prompts = []
        for prompt_context_response in prompt_context_responses:
            replacement_map = {
                "[ANSWER]": prompt_context_response["expected_answers"],
                "[CONTEXT]": prompt_context_response["context"],
                "[CONTENT]": prompt_context_response["model_responses"],
            }
            prompt = self.pb.build_full_prompt(MT_QAC_PROMPT, replacement_map)

            eval_prompts.append({"eval_prompt": prompt, **prompt_context_response})

        return eval_prompts
