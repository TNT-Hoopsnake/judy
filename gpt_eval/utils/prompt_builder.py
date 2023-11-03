from typing import Dict, List
from gpt_eval.config import MetricConfig, TaskTypes
from .prompts import BASE_PROMPT


class PromptBuilder:
    def __init__(self, task_type: TaskTypes, metric_configs: List[MetricConfig]):
        self.task_type = task_type
        self.metric_configs = metric_configs

    @property
    def base_prompt(self):
        metric_descriptions = []
        metric_formats = []
        for metric in self.metric_configs:
            metric_descriptions.append(
                f"\t{metric.name} (Min Score: {metric.score_min}, Max Score: {metric.score_max}): {metric.desc}\n"
            )
            metric_formats.append(f"\t{metric.name}: X\n")

        metric_description = "\n".join(metric_descriptions)
        metric_format = "\n".join(metric_formats)

        prompt = BASE_PROMPT.replace("[METRIC_DESCRIPTIONS]", metric_description)
        prompt = prompt.replace("[METRIC_FORMATS]", metric_format)

        return prompt

    def build_full_prompt(
        self, task_prompt: str, replacement_map: Dict[str, str]
    ) -> str:
        prompt = self.base_prompt
        prompt += task_prompt
        for replace_key, replace_val in replacement_map.items():
            prompt = prompt.replace(replace_key, replace_val)

        return prompt
