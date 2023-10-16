from pydantic import (
    BaseModel, 
    HttpUrl, 
    PositiveInt, 
    confloat, 
    conlist, 
    ConfigDict,
    Field,
    validator
)
from typing import List, Optional

from .constants import (
    ScenarioTypes, 
    ApiTypes
)

# adding "= Field(default=None)" to fields in the following models
# enables those models to be valid while missing those fields entirely
# this is done for most optional fields as well as fields that can be overridden

class ScenarioConfig(BaseModel):
    type: ScenarioTypes
    datasets: List[str]
    model_config = ConfigDict(use_enum_values=True)


class EvaluatedModel(BaseModel):
    name: str
    api_type: ApiTypes
    api_base: HttpUrl

    max_tokens: Optional[PositiveInt] = Field(default=None)
    context_char_limit: Optional[PositiveInt] = Field(default=None)
    temperature: Optional[confloat(ge=0.0, le=2.0)] = Field(default=None)

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
    judge_temperature: confloat(ge=0.0, le=2.0)
    use_proxy: bool

    judge_api_key: Optional[str] = Field(default=None)
    proxies: Optional[Proxies] = Field(default=None)

    @validator('proxies', always=True)
    def validate_proxies(cls, value, values):
        use_proxy = values.get('use_proxy', False)
        if use_proxy and value is None:
            raise ValueError("Proxies must be provided if use_proxy is True")
        return value


class DatasetConfig(BaseModel):
    name: str
    source: HttpUrl
    scenarios: conlist(ScenarioTypes, min_length=1)
    formatter: str

    version: Optional[str] = Field(default=None)

    model_config = ConfigDict(use_enum_values=True)