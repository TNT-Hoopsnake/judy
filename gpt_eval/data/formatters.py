import numpy as np

def xsum_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train, test, validation
    docs_to_summarize = [dataset[ds_split]['document'][i] for i in eval_idxs]
    docs_gt = [dataset[ds_split]['summary'][i] for i in eval_idxs]

    return (docs_to_summarize, docs_gt)


def cnn_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train, test, validation
    docs_to_summarize = [dataset[ds_split]['article'][i] for i in eval_idxs]
    docs_gt = [dataset[ds_split]['highlights'][i] for i in eval_idxs]

    return (docs_to_summarize, docs_gt)


def msmarco_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train, test, validation
    questions = []
    answers = []
    contexts = []

    for i in eval_idxs:
        questions.append(dataset[ds_split]['query'][i])
        # determine the passage that contains the context for the expected answer
        selected_idx = dataset[ds_split]['passages'][i]['is_selected'].index(1)
        contexts.append(dataset[ds_split]['passages'][i]['passage_text'][selected_idx])
        answers.append(dataset[ds_split]['answers'][i][0])

    return (questions, answers, contexts)


def squad_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train, validation
    answers = []
    questions = []
    contexts = []
    for idx in eval_idxs:
        answer = dataset[ds_split]['answers'][idx]['text']
        # exclude questions that do not include answers
        while not answer:
            idx = np.random.randint(0, len(dataset[ds_split]))
            answer = dataset[ds_split]['answers'][idx]['text']
        
        answer = answer[0]
        question = dataset[ds_split]['question'][idx]
        context = dataset[ds_split]['context'][idx]

        questions.append(question)
        answers.append(answer)
        contexts.append(context)

    return (questions, answers, contexts)

def mtbench_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train
    questions = [dataset[ds_split]['turns'][i] for i in eval_idxs]

    return (questions)


def quac_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train, validation

    # retrieve a maximum of 2 questions per context 
    # could add option for including more. we're mostly limited by context size here
    questions = [dataset[ds_split]['questions'][i][:2] for i in eval_idxs]
    answers = [dataset[ds_split]['answers'][i]['texts'][:2] for i in eval_idxs]
    contexts = [dataset[ds_split]['background'][i] for i in eval_idxs]

    return (questions, answers, contexts)


def disinfo_wedging_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train
    groups = [dataset[ds_split]["targeted_group"][i] for i in eval_idxs]
    goals = [dataset[ds_split]["targeted_action"][i] for i in eval_idxs]
    contexts = [dataset[ds_split]["prompt"][i] for i in eval_idxs]
    return (groups, goals, contexts)

def disinfo_reiteration_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train
    thesis = [dataset[ds_split]["thesis"][i] for i in eval_idxs]
    contexts = []
    for idx in eval_idxs:
        context = ''
        for headline in dataset[ds_split]["headlines"][idx]:
            context += f'headline: {headline}\n'
        contexts.append(context)
    return (thesis, contexts)


def open_question_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train, test, validation
    questions = [dataset[ds_split]['question'][i] for i in eval_idxs]

    return questions

def mrqa_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train, test, validation
    questions = [dataset[ds_split]['question'][i] for i in eval_idxs]
    answers = [dataset[ds_split]['answer'][i] for i in eval_idxs]
    contexts = [dataset[ds_split]['context'][i] for i in eval_idxs]

    return (questions, answers, contexts)


def riddle_sense_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train, test, validation
    questions = [dataset[ds_split]['question'][i] for i in eval_idxs]
    answers = []
    for i in eval_idxs:
        answer_key = dataset[ds_split]['answerKey'][i]
        labels = dataset[ds_split]['choices'][i]['label']
        answer_idx = labels.index(answer_key)
        answer =  dataset[ds_split]['choices'][i]['text'][answer_idx]
        answers.append(answer)
    
    return questions, answers

def ethics_suite_formatter(dataset, eval_idxs, ds_split):
    # available splits:
    # train
    questions = [dataset[ds_split]['text'][i] for i in eval_idxs]

    return questions