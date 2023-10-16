import os
from enum import Enum

RESULTS_DIR = os.path.abspath('./results')
DATASETS_DIR = os.path.abspath('./gpt_eval/data/datasets')


DATASET_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'dataset_config.json')
SYSTEM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'system_config.json')
EVAL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'eval_config.json')


class ApiTypes(str, Enum):
    OPENAI="openai"
    TGI="tgi"


class ScenarioTypes(str, Enum):
    SUMMARIZATION="summ"
    MT_QUESTION="mt_q"
    ST_QUESTION_ANSWER_CONTEXT="st_qac"
    MT_QUESTION_ANSWER_CONTEXT="mt_qac"
    DISINFO_WEDGING="disinfo_wedging"
    DISINFO_REITERATION="disinfo_reiteration"

class SourceTypes(str, Enum):
    HUGGINGFACE_HUB="hub"
    URL="url"

def get_responder_class_map():
    # avoid circular dependencies
    from gpt_eval.responders import (
        MTQuestionResponder, 
        SummarizationResponder,
        STQuestionAnswerContextResponder,
        MTQuestionAnswerContextResponder,
        DisinfoReiterationResponder,
        DisinfoWedgingResponder
    )

    return {
        ScenarioTypes.MT_QUESTION:MTQuestionResponder,
        ScenarioTypes.MT_QUESTION_ANSWER_CONTEXT:MTQuestionAnswerContextResponder,
        ScenarioTypes.ST_QUESTION_ANSWER_CONTEXT:STQuestionAnswerContextResponder,
        ScenarioTypes.SUMMARIZATION:SummarizationResponder,
        ScenarioTypes.DISINFO_REITERATION:DisinfoReiterationResponder,
        ScenarioTypes.DISINFO_WEDGING:DisinfoWedgingResponder
    }


def get_config_definitions():
    # avoid circular dependencies
    from gpt_eval.config.config_models import (
        SystemConfig,
        EvaluationConfig,
        DatasetConfig
    )
    return [
        {
            'cls':SystemConfig,
            'path':SYSTEM_CONFIG_PATH,
            'is_list':False,
            'key':'system'
        },
        {
            'cls':EvaluationConfig,
            'path':EVAL_CONFIG_PATH,
            'is_list':False,
            'key':'eval'
        },
        {
            'cls':DatasetConfig,
            'path':DATASET_CONFIG_PATH,
            'is_list':True,
            'key':'datasets'
        }
    ]

JUDGE_CRITERIA = {
    "Accuracy": 0,
    "Coherence": 1,
    "Factuality": 2,
    "Completeness": 3,
    "Relevance": 4,
    "Depth": 5,
    "Creativity": 6,
    "Level of Detail": 7,
}