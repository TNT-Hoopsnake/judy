from .base_responder import BaseResponder
import copy
from gpt_eval.utils.prompts import ST_QAC_PROMPT

# maximum number of characters that will be retrieved from a dataset instance, per prompt
MAX_CONTEXT_SIZE = 1000
SINGLE_TURN_QUESTION_ANSWER_CONTEXT_PREPROMPT = "Using the context above, answer the following question:\n"

class STQuestionAnswerContextResponder(BaseResponder):
    def build_model_prompts(self, formatted_data):
        questions, answers, contexts = formatted_data

        model_prompts = []
        for question, answer, context in zip(questions, answers, contexts):
            append_char = ''
            if not question.endswith('?'):
                append_char = '?'
            prompt = f"[CONTEXT]: {context}\n{SINGLE_TURN_QUESTION_ANSWER_CONTEXT_PREPROMPT}{question}{append_char}"
            model_prompts.append({
                'question':question,
                'prompt':prompt,
                'context':context,
                'gt_answer': answer
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
            base_prompt = copy.copy(ST_QAC_PROMPT)
            
            prompt = base_prompt.replace('[QUESTION]', prompt_context_response['question'])
            prompt = prompt.replace('[ANSWER]', prompt_context_response['response'])
            prompt = prompt.replace('[CONTEXT]', prompt_context_response['context'])

            eval_prompts.append({
                'eval_prompt':prompt,
                **prompt_context_response
            })

        return eval_prompts
