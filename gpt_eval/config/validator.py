from pydantic import (
    BaseModel, 
    HttpUrl, 
    PositiveInt, 
    confloat, 
    conlist, 
    ConfigDict,
    TypeAdapter
)
from typing import List, Optional
import json
import os
from enum import Enum

DATASET_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'dataset_config.json')
SYSTEM_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'system_config.json')
EVAL_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'eval_config.json')


class API_TYPES(str, Enum):
    openai="openai"
    tgi="tgi"

class SCENARIO_TYPES(str, Enum):
    summarization="summ"
    mt_question="mt_q"
    st_question_answer_context="st_qac"
    mt_question_answer_context="mt_qac"


class ScenarioConfig(BaseModel):
    type: SCENARIO_TYPES
    datasets: List[str]

    model_config = ConfigDict(use_enum_values=True)


class EvaluatedModel(BaseModel):
    name: str
    api_type: API_TYPES
    api_base: HttpUrl
    max_tokens: Optional[PositiveInt]
    context_char_limit: Optional[PositiveInt]
    temperature: Optional[confloat(ge=0.0, le=2.0)]

    model_config = ConfigDict(use_enum_values=True)


class EvaluationConfig(BaseModel):
    random_seed: Optional[int]
    num_evals: PositiveInt
    scenarios: conlist(ScenarioConfig, min_length=1)
    max_tokens: PositiveInt
    context_char_limit: PositiveInt
    temperature: confloat(ge=0.0, le=2.0)
    evaluated_models: conlist(EvaluatedModel, min_length=1)


class Proxies(BaseModel):
    http: HttpUrl
    https: HttpUrl

class SystemConfig(BaseModel):
    judge: str
    judge_api_key: Optional[str]
    judge_temperature: confloat(ge=0.0, le=2.0)
    use_proxy: bool
    proxies: Optional[Proxies]


class DatasetConfig(BaseModel):
    name: str
    source: HttpUrl
    version: Optional[str]
    scenarios: conlist(SCENARIO_TYPES, min_length=1)
    formatter: str

    model_config = ConfigDict(use_enum_values=True)


def load_json_from_file(filepath):
    with open(filepath, 'r') as fn:
        json_data = json.load(fn)

    return json_data


def get_validated_data(data, cls):
    validated_data = cls(**data)
    return validated_data


def get_validated_list(data, cls):
    validated_list = TypeAdapter(List[cls]).validate_python(data)
    return validated_list


def load_validated_config(filepath, validate_cls, is_list=False):
    json_data = load_json_from_file(filepath)
    if is_list:
        validated_data = get_validated_list(json_data, validate_cls)
    else:
        validated_data = get_validated_data(json_data, validate_cls)
    return validated_data

        