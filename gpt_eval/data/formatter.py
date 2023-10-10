import numpy as np

def xsum_formatted_data(dataset, eval_idxs):
    docs_to_summarize = [dataset['train']['document'][i] for i in eval_idxs]
    docs_gt = [dataset['train']['summary'][i] for i in eval_idxs]

    return (docs_to_summarize, docs_gt)


def cnn_formatted_data(dataset, eval_idxs):
    docs_to_summarize = [dataset['train']['article'][i] for i in eval_idxs]
    docs_gt = [dataset['train']['highlights'][i] for i in eval_idxs]

    return (docs_to_summarize, docs_gt)


def msmarco_formatted_data(dataset, eval_idxs):
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


def squad_formatted_data(dataset, eval_idxs):
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

def mtbench_formatted_data(dataset, eval_idxs):
    questions = [dataset['train']['turns'][i] for i in eval_idxs]

    return (questions)


def quac_formatted_data(dataset, eval_idxs):
    # retrieve a maximum of 2 questions per context 
    # could add option for including more. we're mostly limited by context size here
    questions = [dataset['train']['questions'][i][:2] for i in eval_idxs]
    answers = [dataset['train']['answers'][i]['texts'][:2] for i in eval_idxs]
    contexts = [dataset['train']['background'][i] for i in eval_idxs]

    return (questions, answers, contexts)

DATASET_MAPPER = {
    'cnn_dailymail':cnn_formatted_data,
    'xsum': xsum_formatted_data,
    'squad_v2':squad_formatted_data,
    'ms_marco':msmarco_formatted_data,
    'mt_bench_en':mtbench_formatted_data,
    'quac':quac_formatted_data
}