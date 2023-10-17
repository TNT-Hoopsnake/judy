from .base_responder import BaseResponder
import copy
from gpt_eval.utils.prompts import MT_Q_PROMPT

class MTQuestionResponder(BaseResponder):
    # the data doesnt need adjusting but for consistency, this function is still used
    def build_model_prompts(self):
        return [{'questions':question} for question in self.data]

    def get_model_responses(self, prompt_questions):
        model_responses = []
        for turns in prompt_questions:
            model_qa = []
            messages = [
                {   
                    'role':'system',
                    'content':'You are a helpful assistant that answers user questions truthfully and to the best of your ability. If you are unable to truthfully respond to a question, simply respond with "CANNOTANSWER"'
                }
            ]
            for turn in turns['questions']:
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

            model_responses.append({
                'model_response':'\n'.join(model_qa),
                **turns
            })

        return model_responses
    
    def build_eval_prompts(self, prompt_responses):
        eval_prompts = []
        for prompt_response in prompt_responses:
            replacement_map = {
                '[CONTENT]':prompt_response['model_response']
            }
            
            prompt = self.pb.build_full_prompt(MT_Q_PROMPT, replacement_map)
 
            eval_prompts.append({
                'eval_prompt':prompt,
                **prompt_response
            })

        return eval_prompts
