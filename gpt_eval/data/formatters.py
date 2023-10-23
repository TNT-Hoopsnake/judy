import numpy as np


def xsum_formatter(dataset, eval_idxs):
    # available splits:
    # train, test, validation
    docs_to_summarize = [dataset["document"][i] for i in eval_idxs]
    docs_gt = [dataset["summary"][i] for i in eval_idxs]

    return (docs_to_summarize, docs_gt)


def cnn_formatter(dataset, eval_idxs):
    # available splits:
    # train, test, validation
    docs_to_summarize = [dataset["article"][i] for i in eval_idxs]
    docs_gt = [dataset["highlights"][i] for i in eval_idxs]

    return (docs_to_summarize, docs_gt)


def msmarco_formatter(dataset, eval_idxs):
    # available splits:
    # train, test, validation
    questions = []
    answers = []
    contexts = []

    for i in eval_idxs:
        questions.append(dataset["query"][i])
        # determine the passage that contains the context for the expected answer
        selected_idx = dataset["passages"][i]["is_selected"].index(1)
        contexts.append(dataset["passages"][i]["passage_text"][selected_idx])
        answers.append(dataset["answers"][i][0])

    return (questions, answers, contexts)


def squad_formatter(dataset, eval_idxs):
    # available splits:
    # train, validation
    answers = []
    questions = []
    contexts = []
    for idx in eval_idxs:
        answer = dataset["answers"][idx]["text"]
        # exclude questions that do not include answers
        while not answer:
            idx = np.random.randint(0, len(dataset))
            answer = dataset["answers"][idx]["text"]

        answer = answer[0]
        question = dataset["question"][idx]
        context = dataset["context"][idx]

        questions.append(question)
        answers.append(answer)
        contexts.append(context)

    return (questions, answers, contexts)


def mtbench_formatter(dataset, eval_idxs):
    # available splits:
    # train
    questions = [dataset["turns"][i] for i in eval_idxs]

    return questions


def quac_formatter(dataset, eval_idxs):
    # available splits:
    # train, validation

    # retrieve a maximum of 2 questions per context
    # could add option for including more. we're mostly limited by context size here
    questions = [dataset["questions"][i][:2] for i in eval_idxs]
    answers = [dataset["answers"][i]["texts"][:2] for i in eval_idxs]
    contexts = [dataset["background"][i] for i in eval_idxs]

    return (questions, answers, contexts)


def disinfo_wedging_formatter(dataset, eval_idxs):
    # available splits:
    # train
    groups = [dataset["targeted_group"][i] for i in eval_idxs]
    goals = [dataset["targeted_action"][i] for i in eval_idxs]
    contexts = [dataset["prompt"][i] for i in eval_idxs]
    return (groups, goals, contexts)


def disinfo_reiteration_formatter(dataset, eval_idxs):
    # available splits:
    # train
    thesis = [dataset["thesis"][i] for i in eval_idxs]
    contexts = []
    for idx in eval_idxs:
        contexts.append(
            "\n".join(
                [f"headline: {headline}" for headline in dataset["headlines"][idx]]
            )
        )
    return (thesis, contexts)


def open_question_formatter(dataset, eval_idxs):
    # available splits:
    # train, test, validation
    questions = [dataset["question"][i] for i in eval_idxs]

    return questions


def mrqa_formatter(dataset, eval_idxs):
    # available splits:
    # train, test, validation
    questions = [dataset["question"][i] for i in eval_idxs]
    answers = [dataset["answer"][i] for i in eval_idxs]
    contexts = [dataset["context"][i] for i in eval_idxs]

    return (questions, answers, contexts)


def riddle_sense_formatter(dataset, eval_idxs):
    # available splits:
    # train, test, validation
    questions = [dataset["question"][i] for i in eval_idxs]
    answers = []
    for i in eval_idxs:
        answer_key = dataset["answerKey"][i]
        labels = dataset["choices"][i]["label"]
        answer_idx = labels.index(answer_key)
        answer = dataset["choices"][i]["text"][answer_idx]
        answers.append(answer)

    return questions, answers


def ethics_suite_formatter(dataset, eval_idxs):
    # available splits:
    # train
    questions = [dataset["text"][i] for i in eval_idxs]

    return questions
