# Adding a Dataset for Evaluation

1. **Choose a Dataset:**
- Identify a dataset suitable for evaluation. For this example, we'll use the GSM8K dataset, a question-answer dataset with 8.5k linguistically diverse grade school math problems, accessible at [GSM8K Dataset](https://huggingface.co/datasets/gsm8k).

2. **Update Dataset Configuration:**
- Add the chosen dataset to your dataset configuration file. You can achieve this using Judy config commands or manually edit the file with your preferred text editor. Run `judy config` to view and modify the default config files.

- The following YAML code is used to define the GSM8K dataset in our dataset configuration file:

```yaml
- id: gsm8k
  source: https://huggingface.co/datasets/gsm8k
  version: main
  tasks: [st_qa]
  formatter: GSM8KFormatter
  split: test
  tags: [maths]
  source_type: hub
```

Breakdown of key fields:
- `id`: Used for loading the dataset with the Hugging Face dataset library.
- `source`: Specifies the URL the dataset was taken from.
- `version`: Specifies the dataset version. This isn't required for all datasets but GSM8K has two possible versions (main or socratic) so we need to specify it.
- `tasks`: Defines the dataset task (e.g., "st_qa" for Single Turn Question Answer).
- `split`: Specifies the dataset split (e.g., test).
- `tags`: Allows you to add tags for filtering during runs.

3. **Formatter Implementation:**
- We specified `GSM8KFormatter` as the dataset formatter in the dataset configuration, so we need to implement it in the `judy/datasets/formatters.py` file:
- Our implementation of GSM8K uses the Single Turn Question Answer (st_qa) task, so we need to ensure the formatter returns data as a `STQAFormattedData` model

```python
class GSM8KFormatter(BaseFormatter):
        def format(self) -> STQAFormattedData:
                questions = []
                answers = []
                for i in self.eval_idxs:
                        questions.append(self.dataset["question"][i])
                        answers.append(self.dataset["answer"][i])

                return STQAFormattedData(
                        questions=questions,
                        answers=answers
                )
```

4. **Add Dataset to Scenario in `eval_config.yaml`:**
- Your new dataset can be used with any scenario of your choice. Add the dataset to one or more scenarios in the `eval_config.yaml` file. For our example, we will add the `gsm8k` dataset to the `Question Answering` scenario:

```yaml
- name: Question Answering
  id: qa
  desc: This scenario tests the model's ability to answer questions under different conditions.
  score_min: 0
  score_max: 10
  datasets: [..., gsm8k]
...
```

5. **Add Scenario to `run_config.yaml`:**
- Update your `run_config.yaml` file to include the scenario that uses the new dataset:
- You can edit this file manually or run `judy config`

```yaml
scenarios: ["qa", ...]
```

6. **Run Evaluation:**
- Execute the evaluation using the following command.
- Update the config file paths and results output directory as necessary. Omitting config file paths means Judy will use default locations (`~/.config/judy/...`). 
- The default locations for your config files can be found by running `judy config`

```bash
judy run -o ./results -dt maths
```

Explanation of parameters:
- `-o`: Specifies the output directory for the evaluation results (e.g., `./results`).
- `-dt`: Includes only datasets with the specified tag (e.g., `maths`).


7. **Summarize Pending Run:**
- After running the command, Judy will provide a summary of your pending run, including evaluations, models, scenarios, and datasets. Example:

```bash
                Welcome to Judy! 

Your pending run will include:

        1 evaluations
        1 models:
                flan-t5-small
        1 scenarios:
                Question Answering
        1 datasets:
                gsm8k
```

- If it all looks good, confirm your run by entering `y`.

That concludes the process! Your dataset should now be loaded, and Judy is ready to commence the evaluation.
