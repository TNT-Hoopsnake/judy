import pathlib
from enum import Enum
from typing import Optional
from platformdirs import user_config_dir, user_cache_dir, user_data_dir

REQUEST_RETRY_MAX_ATTEMPTS = 3
REQUEST_RETRY_WAIT_TIME = 10
REQUEST_RETRY_BACKOFF = 2

APP_NAME = "gpt-eval"
APP_AUTHOR = "tnt"
USER_CONFIG_DIR = pathlib.Path(user_config_dir(APP_NAME))
USER_CACHE_DIR = pathlib.Path(user_cache_dir(APP_NAME, APP_AUTHOR))
DATASETS_DIR = pathlib.Path(user_data_dir(APP_NAME, APP_AUTHOR))

DATASET_CONFIG_PATH = USER_CONFIG_DIR / "dataset_config.yaml"
EVAL_CONFIG_PATH = USER_CONFIG_DIR / "eval_config.yaml"
METRIC_CONFIG_PATH = USER_CONFIG_DIR / "metric_config.yaml"


class ApiTypes(str, Enum):
    OPENAI = "openai"
    TGI = "tgi"


class ScenarioTypes(str, Enum):
    SUMMARIZATION = "summ"
    MT_QUESTION = "mt_q"
    ST_QUESTION = "st_q"
    ST_QUESTION_ANSWER = "st_qa"
    ST_QUESTION_ANSWER_CONTEXT = "st_qac"
    MT_QUESTION_ANSWER_CONTEXT = "mt_qac"
    DISINFO_WEDGING = "disinfo_wedging"
    DISINFO_REITERATION = "disinfo_reiteration"


class SourceTypes(str, Enum):
    HUGGINGFACE_HUB = "hub"
    URL = "url"


class DatasetSplits(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class ModelFamilyTypes(str, Enum):
    LLAMA2 = "llama2"
    FALCON = "chatml_falcon"
    STARCHAT = "chatml_starchat"
    OPEN_ASSISTANT = "open_assistant"
    STABLE_BELUGA = "stablebeluga"
    VICUNA = "vicuna"
    WIZARD = "wizardlm"
    GENERIC = "generic"


class JudgeModels(str, Enum):
    GPT4 = "gpt-4"
    GPT35 = "gpt-3.5-turbo"


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
        STQuestionAnswerResponder,
    )

    return {
        ScenarioTypes.MT_QUESTION: MTQuestionResponder,
        ScenarioTypes.MT_QUESTION_ANSWER_CONTEXT: MTQuestionAnswerContextResponder,
        ScenarioTypes.ST_QUESTION_ANSWER_CONTEXT: STQuestionAnswerContextResponder,
        ScenarioTypes.SUMMARIZATION: SummarizationResponder,
        ScenarioTypes.DISINFO_REITERATION: DisinfoReiterationResponder,
        ScenarioTypes.DISINFO_WEDGING: DisinfoWedgingResponder,
        ScenarioTypes.ST_QUESTION: STQuestionResponder,
        ScenarioTypes.ST_QUESTION_ANSWER: STQuestionAnswerResponder,
    }


def get_config_definitions(
    eval_config: Optional[pathlib.Path],
    dataset_config: Optional[pathlib.Path],
    metric_config: Optional[pathlib.Path],
):
    # avoid circular dependencies
    from gpt_eval.config.config_models import (
        EvaluationConfig,
        DatasetConfig,
        MetricGroupConfig,
    )

    return [
        {
            "cls": EvaluationConfig,
            "path": eval_config or EVAL_CONFIG_PATH,
            "is_list": False,
            "key": "eval",
        },
        {
            "cls": DatasetConfig,
            "path": dataset_config or DATASET_CONFIG_PATH,
            "is_list": True,
            "key": "datasets",
        },
        {
            "cls": MetricGroupConfig,
            "path": metric_config or METRIC_CONFIG_PATH,
            "is_list": True,
            "key": "metrics",
        },
    ]
