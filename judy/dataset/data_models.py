from typing import List
from pydantic import BaseModel


class BaseFormattedData(BaseModel):
    pass


class FormattedData(BaseFormattedData):
    questions: List[str]


class MTQFormattedData(BaseFormattedData):
    questions: List[List[str]]


class MTQACFormattedData(MTQFormattedData):
    contexts: List[str]
    answers: List[List[List[str]]]


class STQAFormattedData(FormattedData):
    answers: List[str]


class STQACFormattedData(STQAFormattedData):
    contexts: List[str]


class STQAMFormattedData(STQAFormattedData):
    metrics: List[List[str]]


class DisinfoWedgingFormattedData(BaseFormattedData):
    groups: List[str]
    goals: List[str]
    contexts: List[str]


class DisinfoReiterationFormattedData(BaseFormattedData):
    thesis: List[str]
    contexts: List[str]


class SummarizationFormattedData(BaseFormattedData):
    docs: List[str]
    answers: List[str]
