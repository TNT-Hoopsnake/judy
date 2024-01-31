# Judy

Judy is a python library and framework to evaluate the text-generation capabilities of Large Language Models (LLM) using a Judge LLM.

Judy allows users to evaluate LLMs using a competent Judge LLM (such as GPT-4). Users can choose from a set of predefined scenarios sourced from recent research, or design their own. A scenario is a specific test designed to evaluate a particular aspect of an LLM. A scenario consists of:

- `Dataset`: A source dataset to generate prompts to evaluate models against.
- `Task`: A task to evaluate models on. Tasks for judge evaluations have been carefully designed by researchers to assess certain aspects of LLMs.
- `Metric`: The metric(s) to use when evaluating the responses from a task. For example - accuracy, level of detail etc.

![Framework Overview](<assets/framework.svg>)

Judy has been inspired by techniques used in research including HELM [1] and LLM-as-a-judge [2].

---
* [1] Holistic Evaluation of Language Models - https://arxiv.org/abs/2211.09110
* [2] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena - https://arxiv.org/abs/2306.05685


## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Judy.

```bash
pip install git+https://github.com/TNT-Hoopsnake/judy
```

## Getting Started

### Setup configs

Judy uses 3 configuration files during evaluation. Only the run config is strictly necessary to begin with:

- `Dataset Config`: Defines all of the datasets available to use in the evaluation run, how to download them and which class to use to format them. **You don't have to worry about specifying this config unless you plan on adding new datasets**. `Judy` will automatically use the example dataset config [here](./judy/config/files/example_dataset_config.yaml) unless you specify an alternate one using `--dataset-config`.
- `Evaluation Config`: Defines all of the tasks and the metrics used to evaluate them. It also restricts which datasets and metrics can be used for each task. **You don't have to worry about specifying this config unless you plan on adding new tasks or metrics**. `Judy` will automatically use the example eval config [here](./judy/config/files/example_eval_config.yaml) unless you specify an alternate one using `--eval-config`.
- `Run Config`: Defines all of the settings to use for your evaluation run. The evaluation results for your run will store a copy (with sensitive details redacted) of these settings as metadata. An example run config is provided [here](./judy/config/files/example_run_config.yaml)

#### Setup model(s) to evaluate

Ensure you have API access to the models you wish to evaluate. We currently support two types of API formats:

* `OPENAI`: The OpenAI API ChatCompletion endpoint ([ref](https://platform.openai.com/docs/api-reference/chat/object))
* `HUGGINGFACE`: The HuggingFace Hosted Inference API ([ref](https://huggingface.co/docs/inference-endpoints/api_reference))

If you are hosting models locally you can use a package like [LocalAI](https://github.com/mudler/LocalAI) to get an OpenAI compatible REST API which can be used by `Judy`.

#### Judy Commands

A CLI interface is provided for viewing and editing Judy config files.

```bash
judy config
```

Run an evaluation as follows:

```bash
judy run --run-config run_config.yml --name disinfo-test --output ./results
```

After running an evaluation, you can serve a web app for viewing the results:

```bash
judy serve -r ./results
```

## Web App Screenshots

The web app allows you to view your evaluation results.

|   |   |
|---|---|
![Overview](<assets/app_home.png>) |  ![App Runs](<assets/app_runs.png>)
![Raw Results](<assets/app_raw.png>)

## Roadmap

### Features

- [x] Core framework
- [x] Web app - to view evaluation results
- [ ] Add perturbations - the ability to modify input datasets - with typos, synonymns etc.
- [ ] Add adaptations - the ability to use different prompting techniques - such as Chain of Thought etc.

### Scenarios

- [x] [FLASK](https://arxiv.org/abs/2307.10928)
- [ ] [Code Comprehension](https://arxiv.org/abs/2308.01240)
- [ ] [Emotional Intelligence](https://arxiv.org/abs/2307.09042)
- [ ] [RED-INSTRUCT](https://arxiv.org/abs/2308.09662)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)