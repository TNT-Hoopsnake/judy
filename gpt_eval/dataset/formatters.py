import numpy as np
from .base import (
    BaseFormatter,
)
from .data_models import (
    MTQFormattedData,
    MTQACFormattedData,
    FormattedData,
    STQAFormattedData,
    STQACFormattedData,
    DisinfoReiterationFormattedData,
    DisinfoWedgingFormattedData,
    SummarizationFormattedData,
)


class XsumFormatter(BaseFormatter):
    def format(self) -> SummarizationFormattedData:
        # available splits:
        # train, test, validation
        docs_to_summarize = [self.dataset["document"][i] for i in self.eval_idxs]
        docs_gt = [self.dataset["summary"][i] for i in self.eval_idxs]

        return SummarizationFormattedData(docs=docs_to_summarize, answers=docs_gt)


class CNNFormatter(BaseFormatter):
    def format(self) -> SummarizationFormattedData:
        # available splits:
        # train, test, validation
        docs_to_summarize = [self.dataset["article"][i] for i in self.eval_idxs]
        docs_gt = [self.dataset["highlights"][i] for i in self.eval_idxs]

        return SummarizationFormattedData(docs=docs_to_summarize, answers=docs_gt)


class MsMarcoFormatter(BaseFormatter):
    def format(self) -> STQACFormattedData:
        # available splits:
        # train, test, validation
        questions = []
        answers = []
        contexts = []

        for i in self.eval_idxs:
            questions.append(self.dataset["query"][i])
            # determine the passage that contains the context for the expected answer
            selected_idx = self.dataset["passages"][i]["is_selected"].index(1)
            contexts.append(self.dataset["passages"][i]["passage_text"][selected_idx])
            answers.append(self.dataset["answers"][i][0])

        return STQACFormattedData(
            questions=questions, answers=answers, contexts=contexts
        )


class SquadFormatter(BaseFormatter):
    def format(self) -> STQACFormattedData:
        # available splits:
        # train, validation
        answers = []
        questions = []
        contexts = []
        for idx in self.eval_idxs:
            answer = self.dataset["answers"][idx]["text"]
            # exclude questions that do not include answers
            while not answer:
                idx = np.random.randint(0, len(self.dataset))
                answer = self.dataset["answers"][idx]["text"]

            answer = answer[0]
            question = self.dataset["question"][idx]
            context = self.dataset["context"][idx]

            questions.append(question)
            answers.append(answer)
            contexts.append(context)

        return STQACFormattedData(
            questions=questions, answers=answers, contexts=contexts
        )


class MTBenchFormatter(BaseFormatter):
    def format(self) -> MTQFormattedData:
        # available splits:
        # train
        questions = [self.dataset["turns"][i] for i in self.eval_idxs]

        return MTQFormattedData(questions=questions)


class QuacFormatter(BaseFormatter):
    def format(self) -> MTQACFormattedData:
        # available splits:
        # train, validation

        # retrieve a maximum of 2 questions per context
        # could add option for including more. we're mostly limited by context size here
        questions = [self.dataset["questions"][i][:2] for i in self.eval_idxs]
        answers = [self.dataset["answers"][i]["texts"][:2] for i in self.eval_idxs]
        contexts = [self.dataset["background"][i] for i in self.eval_idxs]
        return MTQACFormattedData(
            questions=questions, answers=answers, contexts=contexts
        )


class DisinfoWedgingFormatter(BaseFormatter):
    def format(self) -> DisinfoWedgingFormattedData:
        # available splits:
        # train
        groups = [self.dataset["targeted_group"][i] for i in self.eval_idxs]
        goals = [self.dataset["targeted_action"][i] for i in self.eval_idxs]
        contexts = [self.dataset["prompt"][i] for i in self.eval_idxs]
        return DisinfoWedgingFormattedData(
            groups=groups, goals=goals, contexts=contexts
        )


class DisinfoReiterationFormatter(BaseFormatter):
    def format(self) -> DisinfoReiterationFormattedData:
        # available splits:
        # train
        thesis = [self.dataset["thesis"][i] for i in self.eval_idxs]
        contexts = []
        for idx in self.eval_idxs:
            contexts.append(
                "\n".join(
                    [
                        f"headline: {headline}"
                        for headline in self.dataset["headlines"][idx]
                    ]
                )
            )
        return DisinfoReiterationFormattedData(thesis=thesis, contexts=contexts)


class OpenQuestionFormatter(QuacFormatter):
    def format(self) -> FormattedData:
        # available splits:
        # train, test, validation
        questions = [self.dataset["question"][i] for i in self.eval_idxs]

        return FormattedData(questions=questions)


class MRQAFormatter(QuacFormatter):
    def format(self):
        # available splits:
        # train, test, validation
        questions = [self.dataset["question"][i] for i in self.eval_idxs]
        answers = [self.dataset["answer"][i] for i in self.eval_idxs]
        contexts = [self.dataset["context"][i] for i in self.eval_idxs]

        return STQACFormattedData(
            questions=questions, answers=answers, contexts=contexts
        )


class RiddleSenseFormatter(BaseFormatter):
    def format(self) -> STQAFormattedData:
        # available splits:
        # train, test, validation
        questions = [self.dataset["question"][i] for i in self.eval_idxs]
        answers = []
        for i in self.eval_idxs:
            answer_key = self.dataset["answerKey"][i]
            labels = self.dataset["choices"][i]["label"]
            answer_idx = labels.index(answer_key)
            answer = self.dataset["choices"][i]["text"][answer_idx]
            answers.append(answer)
        return STQAFormattedData(questions=questions, answers=answers)


class EthicsSuiteFormatter(BaseFormatter):
    def format(self) -> FormattedData:
        # available splits:
        # train
        questions = [self.dataset["text"][i] for i in self.eval_idxs]

        return FormattedData(questions=questions)
