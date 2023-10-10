from .base_responder import BaseResponder
import copy
from gpt_eval.utils.prompts import MT_Q_PROMPT

class MTQuestionResponder(BaseResponder):
    # the data doesnt need adjusting but for consistency, this function is still used
    def build_model_prompts(self, formatted_data):
        return formatted_data

    def get_model_responses(self, questions):
        model_responses = []
        for turns in questions:
            model_qa = []
            messages = [
                {   
                    'role':'system',
                    'content':'You are a helpful assistant that answers user questions truthfully and to the best of your ability. If you are unable to truthfully respond to a question, simply respond with "CANNOTANSWER"'
                }
            ]
            for turn in turns:
                messages.append({
                    'role':'user',
                    'content':turn
                })
                
                model_ans = self.query_chat_model(messages)
                messages.append({
                    'role':'assistant',
                    'content':model_ans
                })
                model_qa.append(f"[USER QUESTION]{turn}\n[ASSISTANT RESPONSE]{model_ans}")

            model_responses.append('\n'.join(model_qa))

        return model_responses
    
    def build_eval_prompts(self, prompt_responses):
        eval_prompts = []
        for prompt_response in prompt_responses:
            base_prompt = copy.copy(MT_Q_PROMPT)
        
            prompt = base_prompt.replace('[CONTENT]', prompt_response)

            eval_prompts.append({
                'eval_prompt':prompt,
                **prompt_response
            })

        return eval_prompts
