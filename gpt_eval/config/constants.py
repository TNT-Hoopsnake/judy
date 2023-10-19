import os
import pathlib
from enum import Enum

REQUEST_RETRY_MAX_ATTEMPTS = 3
REQUEST_RETRY_WAIT_TIME = 10
REQUEST_RETRY_BACKOFF = 2

RESULTS_DIR = os.path.abspath('./results')

USER_DIR = pathlib.Path.home() / ".gpt-eval"
USER_CONFIG_DIR = USER_DIR / "config"
USER_CACHE_DIR = USER_DIR / "cache"
DATASETS_DIR = USER_DIR / "datasets"

DATASET_CONFIG_PATH = USER_CONFIG_DIR / 'dataset_config.yaml'
SYSTEM_CONFIG_PATH = USER_CONFIG_DIR / 'system_config.yaml'
EVAL_CONFIG_PATH = USER_CONFIG_DIR / 'eval_config.yaml'
METRIC_CONFIG_PATH = USER_CONFIG_DIR / 'metric_config.yaml'

class ApiTypes(str, Enum):
    OPENAI="openai"
    TGI="tgi"


class ScenarioTypes(str, Enum):
    SUMMARIZATION="summ"
    MT_QUESTION="mt_q"
    ST_QUESTION="st_q"
    ST_QUESTION_ANSWER="st_qa"
    ST_QUESTION_ANSWER_CONTEXT="st_qac"
    MT_QUESTION_ANSWER_CONTEXT="mt_qac"
    DISINFO_WEDGING="disinfo_wedging"
    DISINFO_REITERATION="disinfo_reiteration"

class SourceTypes(str, Enum):
    HUGGINGFACE_HUB="hub"
    URL="url"

class DatasetSplits(str, Enum):
    TRAIN="train"
    TEST="test"
    VALIDATION="validation"

class ModelFamilyTypes(str, Enum):
    LLAMA2="llama2"
    FALCON="chatml_falcon"
    STARCHAT="chatml_starchat"
    OPEN_ASSISTANT="open_assistant"
    STABLE_BELUGA="stablebeluga"
    VICUNA="vicuna"
    WIZARD="wizardlm"
    GENERIC="generic"

def get_responder_class_map():
    # avoid circular dependencies
    from gpt_eval.responders import (
        MTQuestionResponder, 
        SummarizationResponder,
        STQuestionAnswerContextResponder,
        MTQuestionAnswerContextResponder,
        DisinfoReiterationResponder,
        DisinfoWedgingResponder,
        STQuestionResponder,
        STQuestionAnswerResponder
    )

    return {
        ScenarioTypes.MT_QUESTION:MTQuestionResponder,
        ScenarioTypes.MT_QUESTION_ANSWER_CONTEXT:MTQuestionAnswerContextResponder,
        ScenarioTypes.ST_QUESTION_ANSWER_CONTEXT:STQuestionAnswerContextResponder,
        ScenarioTypes.SUMMARIZATION:SummarizationResponder,
        ScenarioTypes.DISINFO_REITERATION:DisinfoReiterationResponder,
        ScenarioTypes.DISINFO_WEDGING:DisinfoWedgingResponder,
        ScenarioTypes.ST_QUESTION:STQuestionResponder,
        ScenarioTypes.ST_QUESTION_ANSWER:STQuestionAnswerResponder
    }


def get_config_definitions():
    # avoid circular dependencies
    from gpt_eval.config.config_models import (
        SystemConfig,
        EvaluationConfig,
        DatasetConfig,
        MetricConfig
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
        },
        {
            'cls':MetricConfig,
            'path':METRIC_CONFIG_PATH,
            'is_list':True,
            'key':'metrics'
        }
    ]