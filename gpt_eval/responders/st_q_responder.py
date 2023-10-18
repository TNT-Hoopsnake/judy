from .base_responder import BaseResponder
import copy
from gpt_eval.utils.prompts import ST_Q_PROMPT

SINGLE_TURN_QUESTION_PREPROMPT = "Answer the following question:\n"

class STQuestionResponder(BaseResponder):
    def build_model_prompts(self):
        questions = self.data

        model_prompts = []
        for question in questions:
            prompt = f"{SINGLE_TURN_QUESTION_PREPROMPT}{question}"
            model_prompts.append({
                'question':question,
                'prompt':prompt,
            })

        return model_prompts
    
    def get_model_responses(self, prompt_contexts):
        model_responses = []
        for prompt_context in prompt_contexts:
            response = self.query_model(prompt_context['prompt'])
            # eventually should be linked via sql tables
            model_responses.append({
                'response':response,
                **prompt_context
            })

        return model_responses
    
    def build_eval_prompts(self, prompt_context_responses):
        eval_prompts = []
        for prompt_context_response in prompt_context_responses:
            replacement_map = {
                '[QUESTION]':prompt_context_response['question'],
                '[ANSWER]':prompt_context_response['response'],
            }

            prompt = self.pb.build_full_prompt(ST_Q_PROMPT, replacement_map)
            
            eval_prompts.append({
                'eval_prompt':prompt,
                **prompt_context_response
            })

        return eval_prompts
