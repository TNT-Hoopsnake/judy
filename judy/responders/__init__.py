from judy.config import TaskTypes
from .data_models import (
    ModelPrompt,
    ModelResponse,
    EvalPrompt,
    MetricScore,
    EvalResponse,
)
from .disinfo_reiteration_responder import DisinfoReiterationResponder
from .disinfo_wedging_responder import DisinfoWedgingResponder
from .mt_q_responder import MTQuestionResponder
from .mt_qac_responder import MTQuestionAnswerContextResponder
from .st_q_responder import STQuestionResponder
from .st_qa_responder import STQuestionAnswerResponder
from .st_qac_responder import STQuestionAnswerContextResponder
from .summ_responder import SummarizationResponder


def get_responder_class_map():
    return {
        TaskTypes.MT_QUESTION: MTQuestionResponder,
        TaskTypes.MT_QUESTION_ANSWER_CONTEXT: MTQuestionAnswerContextResponder,
        TaskTypes.ST_QUESTION_ANSWER_CONTEXT: STQuestionAnswerContextResponder,
        TaskTypes.SUMMARIZATION: SummarizationResponder,
        TaskTypes.DISINFO_REITERATION: DisinfoReiterationResponder,
        TaskTypes.DISINFO_WEDGING: DisinfoWedgingResponder,
        TaskTypes.ST_QUESTION: STQuestionResponder,
        TaskTypes.ST_QUESTION_ANSWER: STQuestionAnswerResponder,
    }
