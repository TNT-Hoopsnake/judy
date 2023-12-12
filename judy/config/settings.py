import pathlib
from enum import Enum
from platformdirs import user_config_dir, user_cache_dir, user_data_dir

REQUEST_RETRY_MAX_ATTEMPTS = 3
REQUEST_RETRY_WAIT_TIME = 10
REQUEST_RETRY_BACKOFF = 2

APP_NAME = "judy"
APP_AUTHOR = "tnt"
USER_CONFIG_DIR = pathlib.Path(user_config_dir(APP_NAME))
USER_CACHE_DIR = pathlib.Path(user_cache_dir(APP_NAME, APP_AUTHOR))
DATASETS_DIR = pathlib.Path(user_data_dir(APP_NAME, APP_AUTHOR))

DATASET_CONFIG_PATH = USER_CONFIG_DIR / "dataset_config.yaml"
EVAL_CONFIG_PATH = USER_CONFIG_DIR / "eval_config.yaml"
RUN_CONFIG_PATH = USER_CONFIG_DIR / "run_config.yaml"

MODEL_BATCH_SIZE = 100
JUDGE_BATCH_SIZE = 100

LOG_FILE_PATH = "app.log"

DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"


class ApiTypes(str, Enum):
    OPENAI = "openai"
    TGI = "tgi"


class TaskTypes(str, Enum):
    SUMMARIZATION = "summ"
    MT_QUESTION = "mt_q"
    ST_QUESTION = "st_q"
    ST_QUESTION_ANSWER = "st_qa"
    ST_QUESTION_ANSWER_CONTEXT = "st_qac"
    ST_QUESTION_ANSWER_METRIC = "st_qam"
    MT_QUESTION_ANSWER_CONTEXT = "mt_qac"
    DISINFO_WEDGING = "disinfo_wedging"
    DISINFO_REITERATION = "disinfo_reiteration"


class IgnoreCacheTypes(str, Enum):
    ALL = "all"
    DATASET = "datasets"
    PROMPTS = "prompts"
    EVALS = "evals"


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
    GPT4TURBO = "gpt-4-1106-preview"


class InputTokenCost(float, Enum):
    GPT4 = 0.01 / 1000.0
    GPT35 = 0.001 / 1000.0
    GPT4TURBO = 0.01 / 1000.0


class OutputTokenCost(float, Enum):
    GPT4 = 0.03 / 1000.0
    GPT35 = 0.002 / 1000.0
    GPT4TURBO = 0.03 / 1000.0
