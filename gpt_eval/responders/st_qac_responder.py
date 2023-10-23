from gpt_eval.utils.prompts import ST_QAC_PROMPT
from .base_responder import BaseResponder

SINGLE_TURN_QUESTION_ANSWER_CONTEXT_PREPROMPT = (
    "Using the context above, answer the following question:\n"
)


class STQuestionAnswerContextResponder(BaseResponder):
    def build_model_prompts(self):
        questions, answers, contexts = self.data

        model_prompts = []
        for question, answer, context in zip(questions, answers, contexts):
            context = context[: self._context_char_limit]
            append_char = ""
            if not question.endswith("?"):
                append_char = "?"
            prompt = f"[CONTEXT]: {context}\n{SINGLE_TURN_QUESTION_ANSWER_CONTEXT_PREPROMPT}{question}{append_char}"
            model_prompts.append(
                {
                    "question": question,
                    "prompt": prompt,
                    "context": context,
                    "gt_answer": answer,
                }
            )

        return model_prompts

    def get_model_responses(self, prompt_contexts):
        model_responses = []
        for prompt_context in prompt_contexts:
            response = self.query_model(prompt_context["prompt"])
            # eventually should be linked via sql tables
            model_responses.append({"response": response, **prompt_context})

        return model_responses

    def build_eval_prompts(self, prompt_context_responses):
        eval_prompts = []
        for prompt_context_response in prompt_context_responses:
            replacement_map = {
                "[QUESTION]": prompt_context_response["question"],
                "[ANSWER]": prompt_context_response["response"],
                "[CONTEXT]": prompt_context_response["context"],
            }

            prompt = self.pb.build_full_prompt(ST_QAC_PROMPT, replacement_map)

            eval_prompts.append({"eval_prompt": prompt, **prompt_context_response})

        return eval_prompts
