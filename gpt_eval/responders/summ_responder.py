from gpt_eval.utils.prompts import SUMMARIZATION_PROMPT
from .base_responder import BaseResponder

SUMMARY_PREPROMPT = "Provide a concise and accurate summary of the following text:\n"


class SummarizationResponder(BaseResponder):
    def build_model_prompts(self):
        docs, docs_gt = self.data
        model_prompts = []
        for doc, gt in zip(docs, docs_gt):
            prompt = f'{SUMMARY_PREPROMPT}"{doc[:self._context_char_limit]}"\n'

            model_prompts.append(
                {
                    "prompt": prompt,
                    "context": doc[: self._context_char_limit],
                    "gt_answer": gt,
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
                "[DISCUSSION]": prompt_context_response["context"],
                "[SUMMARY]": prompt_context_response["response"],
            }
            prompt = self.pb.build_full_prompt(SUMMARIZATION_PROMPT, replacement_map)

            eval_prompts.append({"eval_prompt": prompt, **prompt_context_response})

        return eval_prompts
