from typing import List, Optional

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    PositiveInt,
    confloat,
    conlist,
    validator,
)


from .settings import (
    DatasetSplits,
    ModelFamilyTypes,
    TaskTypes,
    SourceTypes,
    JudgeModels,
    ApiTypes,
)

# adding "= Field(default=None)" to fields in the following models
# enables those models to be valid while missing those fields entirely
# Any field that makes use of this must be optional.
# When dumping models to json, these fields will exist but will be null and we must allow for that


class TaskConfig(BaseModel):
    id: TaskTypes
    name: Optional[str] = Field(default=None)
    desc: Optional[str] = Field(default=None)
    tags: Optional[List[str]] = Field(default=None)
    task_preprompt: Optional[str] = Field(default=None)
    eval_prompt: Optional[str] = Field(default=None)


class Proxies(BaseModel):
    http: HttpUrl
    https: HttpUrl


class AuthenticatedLLMModel(BaseModel):
    api_type: ApiTypes
    api_base: HttpUrl | None = Field(default=None)
    api_key: str | None = Field(default=None)
    use_proxy: bool = False
    proxies: Optional[Proxies] = Field(default=None)

    @validator("proxies", always=True)
    def validate_proxies(cls, value, values):  # pylint: disable=no-self-argument
        use_proxy = values.get("use_proxy", False)
        if use_proxy and value is None:
            raise ValueError("Proxies must be provided if use_proxy is True")
        return value

    @validator("api_base", always=True)
    def validate_api_base(cls, v, values):  # pylint: disable=no-self-argument
        if values["api_type"] == ApiTypes.TGI:
            if not v:
                raise ValueError(
                    f"api_base must be defined for api_type: {values['api_type']}"
                )
        return v


class LLMModel(AuthenticatedLLMModel):
    name: Optional[str] = Field(default=None)
    family: ModelFamilyTypes = ModelFamilyTypes.GENERIC

    def __hash__(self):
        return hash((self.name, self.api_type.value, self.api_base or None))


class EvaluatedModel(AuthenticatedLLMModel):
    id: str
    name: Optional[str] = Field(default=None)
    max_tokens: Optional[PositiveInt] = Field(default=None)
    context_char_limit: Optional[PositiveInt] = Field(default=None)
    temperature: Optional[confloat(ge=0.0, le=2.0)] = Field(default=None)
    family: ModelFamilyTypes = ModelFamilyTypes.GENERIC
    tags: Optional[List[str]] = Field(default=None)


class JudgeModel(AuthenticatedLLMModel):
    name: JudgeModels
    max_tokens: Optional[PositiveInt] = Field(default=None)
    temperature: Optional[confloat(ge=0.0, le=2.0)] = Field(default=None)


class RunConfig(BaseModel):
    random_seed: Optional[int]
    num_evals: PositiveInt
    max_tokens: PositiveInt
    context_char_limit: PositiveInt
    temperature: confloat(ge=0.0, le=2.0)
    judge: JudgeModel
    models: conlist(EvaluatedModel, min_length=1)
    scenarios: conlist(str, min_length=1)


class MetricConfig(BaseModel):
    name: str
    desc: str
    score_min: Optional[int] = Field(default=None)
    score_max: Optional[int] = Field(default=None)


class ScenarioConfig(BaseModel):
    id: str
    name: Optional[str] = Field(default=None)
    desc: Optional[str] = Field(default=None)
    score_min: int = 0
    score_max: int = 10
    datasets: conlist(str, min_length=1)
    metrics: conlist(MetricConfig, min_length=1)


class EvaluationConfig(BaseModel):
    tasks: conlist(TaskConfig, min_length=1)
    scenarios: conlist(ScenarioConfig, min_length=1)


class DatasetTask(BaseModel):
    id: TaskTypes
    formatter: str


class DatasetConfig(BaseModel):
    id: str
    name: Optional[str] = Field(default=None)
    source: HttpUrl
    tasks: conlist(DatasetTask, min_length=1)
    source_type: Optional[SourceTypes] = SourceTypes.HUGGINGFACE_HUB
    version: Optional[str] = Field(default=None)
    split: DatasetSplits = DatasetSplits.TRAIN
    tags: Optional[List[str]] = Field(default=None)
