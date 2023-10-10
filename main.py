from gpt_eval.responders import (
    MTQuestionResponder, 
    SummarizationResponder,
    STQuestionAnswerContextResponder,
    MTQuestionAnswerContextResponder
)
from gpt_eval.evaluation import Evaluator
import json

# 'dim/mt_bench_en'
if __name__ == "__main__":
    responder = MTQuestionAnswerContextResponder(
        model='gpt-3-5', 
        ds_name='quac',
        ds_vers=None, 
        api_base='http://localhost:8080', 
        api_key=None
    )


    eval_prompts = responder.get_evaluation_prompts(1)

    print(json.dumps(eval_prompts, indent=4))

    eval_prompts = [a['eval_prompt'] for a in eval_prompts]
    evaluator = Evaluator('flan-t5-small', 'quac', False)
    evaluator.run_evaluation(eval_prompts)
