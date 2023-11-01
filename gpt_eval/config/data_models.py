from typing import List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    PositiveInt,
    confloat,
    conlist,
    validator,
)


from .settings import (
    ApiTypes,
    DatasetSplits,
    ModelFamilyTypes,
    TaskTypes,
    SourceTypes,
    JudgeModels,
)

# adding "= Field(default=None)" to fields in the following models
# enables those models to be valid while missing those fields entirely
# Any field that makes use of this must be optional.
# When dumping models to json, these fields will exist but will be null and we must allow for that


class TaskConfig(BaseModel):
    id: TaskTypes
    name: Optional[str] = Field(default=None)
    desc: Optional[str] = Field(default=None)
    datasets: conlist(str, min_length=1)
    model_config = ConfigDict(use_enum_values=True)
    tags: Optional[List[str]] = Field(default=None)


class EvaluatedModel(BaseModel):
    id: str
    name: Optional[str] = Field(default=None)
    api_type: ApiTypes
    api_base: HttpUrl

    max_tokens: Optional[PositiveInt] = Field(default=None)
    context_char_limit: Optional[PositiveInt] = Field(default=None)
    temperature: Optional[confloat(ge=0.0, le=2.0)] = Field(default=None)
    family: ModelFamilyTypes = ModelFamilyTypes.GENERIC
    tags: Optional[List[str]] = Field(default=None)

    model_config = ConfigDict(use_enum_values=True)


class Proxies(BaseModel):
    http: HttpUrl
    https: HttpUrl


class RunConfig(BaseModel):
    judge: JudgeModels
    judge_temperature: confloat(ge=0.0, le=2.0)
    judge_api_key: Optional[str] = Field(default=None)
    use_proxy: bool
    proxies: Optional[Proxies] = Field(default=None)
    random_seed: Optional[int]
    num_evals: PositiveInt
    max_tokens: PositiveInt
    context_char_limit: PositiveInt
    temperature: confloat(ge=0.0, le=2.0)
    models: conlist(EvaluatedModel, min_length=1)
    tasks: conlist(str, min_length=1)
    metrics: conlist(str, min_length=1)

    @validator("proxies", always=True)
    def validate_proxies(cls, value, values):  # pylint: disable=no-self-argument
        use_proxy = values.get("use_proxy", False)
        if use_proxy and value is None:
            raise ValueError("Proxies must be provided if use_proxy is True")
        return value


class MetricConfig(BaseModel):
    name: str
    desc: str
    tasks: Optional[List[TaskTypes]] = Field(default=None)
    min: Optional[int] = Field(default=None)
    max: Optional[int] = Field(default=None)


class MetricGroupConfig(BaseModel):
    id: str
    name: str = Field(default=None)
    desc: str = Field(default=None)
    tasks: conlist(TaskTypes, min_length=1)
    min: int = 0
    max: int = 10
    metrics: conlist(MetricConfig, min_length=1)


class EvaluationConfig(BaseModel):
    tasks: conlist(TaskConfig, min_length=1)
    metric_groups: conlist(MetricGroupConfig, min_length=1)


class DatasetConfig(BaseModel):
    id: str
    name: Optional[str] = Field(default=None)
    source: HttpUrl
    tasks: conlist(TaskTypes, min_length=1)
    formatter: str
    source_type: Optional[SourceTypes] = SourceTypes.HUGGINGFACE_HUB
    version: Optional[str] = Field(default=None)
    split: DatasetSplits = DatasetSplits.TRAIN
    model_config = ConfigDict(use_enum_values=True)
    tags: Optional[List[str]] = Field(default=None)
