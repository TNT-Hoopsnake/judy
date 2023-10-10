import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('OPENAI_KEY', None)
RANDOM_SEED = int(os.getenv('RANDOM_SEED', 9999))
MODEL_ENDPOINT = os.getenv('MODEL_ENDPOINT', 'http://localhost:8080')
PROXIES = {
    'http':os.getenv('HTTP_PROXY'),
    'https':os.getenv('HTTPS_PROXY')
}

TGI_ENDPOINT = MODEL_ENDPOINT + '/generate'
LLAMA_CPP_ENDPOINT = MODEL_ENDPOINT + '/completion'
DW_ENDPOINT = MODEL_ENDPOINT + f'/prompt/{os.getenv("MODEL_LIB", "tgi")}'

MAX_NEW_TOKENS = 300

RESULTS_DIR = os.path.abspath('./results')
DATASETS_DIR = os.path.abspath('./datasets')
COMBINED_RESULTS_DIR = os.path.abspath('./combined_results')

TGI_QUERY='tgi'
DW_QUERY='dw'
LLAMA_CPP_QUERY='llamacpp'
OPENAI_QUERY="openai"


CRITERIA = {
    "Accuracy": 0,
    "Coherence": 1,
    "Factuality": 2,
    "Completeness": 3,
    "Relevance": 4,
    "Depth": 5,
    "Creativity": 6,
    "Level of Detail": 7,
}

LLAMA_FAMILY = "llama"
ALPACA_FAMILY = "alpaca"
GENERIC_FAMILY = "generic"
ORCA_FAMILY = "orca"

MODEL_FAMILY_MAP = {
    LLAMA_FAMILY: [
        "llama-7b-chat", "llama-70b-chat"
    ],
    ALPACA_FAMILY: [
        "nous-hermes-llama-70b", "lazarus-30b", "platypus-70b", "tulpar-7b", "llama-30b-instruct"
    ],
    GENERIC_FAMILY: [
        "falcon-7b", "falcon-7b-instruct", "falcon-40b", "falcon-40b-instruct", "flan-t5-large", "llama-70b", "llama-7b"
    ],
    ORCA_FAMILY: [
        "llong-orca-7b"
    ],
}

FAMILY_PROMPT_FORMAT = {
    LLAMA_FAMILY:'<s>[INST]<<SYS>><</SYS>> [PROMPT][/INST]',
    ALPACA_FAMILY:'### Instruction:\n[PROMPT]\n\n###Response:\n',
    GENERIC_FAMILY: 'User: [PROMPT]\nAssistant: ',
    ORCA_FAMILY:'<|im_start|>user\n[PROMPT]<|im_end|>\n<|im_start|>assistant\n',
}

def get_model_family(model_name):
    model_family = None
    for family, models in MODEL_FAMILY_MAP.items():
        if model_name in models:
            model_family = family
            break
    # Fuzzy match if there are no exact matches
    if model_family == None:
        for family, models in MODEL_FAMILY_MAP.items():
            if any([model_name.startswith(x) for x in models]):
                model_family = family
                break

    if model_family == None:
        print(f'Unable to determine family for model: {model_name}. Will use generic prompt format')
        model_family = GENERIC_FAMILY

    return model_family



def get_formatted_prompt(text, model_name, prev_responses):
    model_family = get_model_family(model_name)

    prompt_format = FAMILY_PROMPT_FORMAT[model_family]
    additional_prompt = ''
    for quest, prev_resp in prev_responses.items():
        eos_token = ''
        if model_family == LLAMA_FAMILY:
            eos_token = '</s>'
        elif model_family == ORCA_FAMILY:
            eos_token = '<|im_end|>\n'
        formatted_resp = f"{prompt_format.replace('[PROMPT]', quest)}\n{prev_resp}{eos_token}"
        additional_prompt += formatted_resp

    return ' '.join([additional_prompt, prompt_format.replace('[PROMPT]', text)])


def get_est_token_cost(eval_model, num_tokens):
    if eval_model == 'gpt-4':
        cost_factor = (0.03 / 1000) 
    elif eval_model == 'gpt-3.5-turbo':
        cost_factor = (0.0015 / 1000)

    return round(num_tokens * cost_factor, 5)