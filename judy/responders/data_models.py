from typing import List
from pydantic import (
    BaseModel,
)


class ModelPrompt(BaseModel):
    pass


class ModelResponse(BaseModel):
    response: str
    prompt: ModelPrompt


class EvalPrompt(BaseModel):
    prompt: str
    response_data: ModelResponse


class MetricScore(BaseModel):
    name: str
    score: int


class EvalResponse(BaseModel):
    prompt: str
    response: str
    scores: List[MetricScore]
