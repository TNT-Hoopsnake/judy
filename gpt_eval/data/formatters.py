import numpy as np

def xsum_formatter(dataset, eval_idxs):
    docs_to_summarize = [dataset['train']['document'][i] for i in eval_idxs]
    docs_gt = [dataset['train']['summary'][i] for i in eval_idxs]

    return (docs_to_summarize, docs_gt)


def cnn_formatter(dataset, eval_idxs):
    docs_to_summarize = [dataset['train']['article'][i] for i in eval_idxs]
    docs_gt = [dataset['train']['highlights'][i] for i in eval_idxs]

    return (docs_to_summarize, docs_gt)


def msmarco_formatter(dataset, eval_idxs):
    questions = []
    answers = []
    contexts = []

    for i in eval_idxs:
        questions.append(dataset['train']['query'][i])
        # determine the passage that contains the context for the expected answer
        selected_idx = dataset['train']['passages'][i]['is_selected'].index(1)
        contexts.append(dataset['train']['passages'][i]['passage_text'][selected_idx])
        answers.append(dataset['train']['answers'][i][0])

    return (questions, answers, contexts)


def squad_formatter(dataset, eval_idxs):
    answers = []
    questions = []
    contexts = []
    for idx in eval_idxs:
        answer = dataset['train']['answers'][idx]['text']
        # exclude questions that do not include answers
        while not answer:
            idx = np.random.randint(0, len(dataset['train']))
            answer = dataset['train']['answers'][idx]['text']
        
        answer = answer[0]
        question = dataset['train']['question'][idx]
        context = dataset['train']['context'][idx]

        questions.append(question)
        answers.append(answer)
        contexts.append(context)

    return (questions, answers, contexts)

def mtbench_formatter(dataset, eval_idxs):
    questions = [dataset['train']['turns'][i] for i in eval_idxs]

    return (questions)


def quac_formatter(dataset, eval_idxs):
    # retrieve a maximum of 2 questions per context 
    # could add option for including more. we're mostly limited by context size here
    questions = [dataset['train']['questions'][i][:2] for i in eval_idxs]
    answers = [dataset['train']['answers'][i]['texts'][:2] for i in eval_idxs]
    contexts = [dataset['train']['background'][i] for i in eval_idxs]

    return (questions, answers, contexts)


def disinfo_wedging_formatter(dataset, eval_idxs):
    groups = [dataset["train"]["targeted_group"][i] for i in eval_idxs]
    goals = [dataset["train"]["targeted_action"][i] for i in eval_idxs]
    contexts = [dataset["train"]["prompt"][i] for i in eval_idxs]
    return (groups, goals, contexts)

def disinfo_reiteration_formatter(dataset, eval_idxs):
    thesis = [dataset["train"]["thesis"][i] for i in eval_idxs]
    contexts = []
    for idx in eval_idxs:
        context = ''
        for headline in dataset["train"]["headlines"][idx]:
            context += f'headline: {headline}\n'
        contexts.append(context)
    return (thesis, contexts)


def open_question_formatter(dataset, eval_idxs):
    questions = [dataset['train']['question'][i] for i in eval_idxs]

    return questions

def mrqa_formatter(dataset, eval_idxs):
    questions = [dataset['train']['question'][i] for i in eval_idxs]
    answers = [dataset['train']['question'][i] for i in eval_idxs]
    contexts = [dataset['train']['context'][i] for i in eval_idxs]

    return (questions, answers, contexts)


def riddle_sense_formatter(dataset, eval_idxs):
    questions = [dataset['train']['question'][i] for i in eval_idxs]
    answers = []
    for i in eval_idxs:
        answer_key = dataset['train']['answerKey'][i]
        labels = dataset['train']['choices'][i]['label']
        answer_idx = labels.index(answer_key)
        answer =  dataset['train']['choices'][i]['text'][answer_idx]
        answers.append(answer)
    
    return questions, answers
