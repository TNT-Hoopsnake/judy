from typing import Dict, List
from judy.config import MetricConfig, TaskTypes
from .prompts import BASE_PROMPT


class PromptBuilder:
    def __init__(self, task_type: TaskTypes, metric_configs: List[MetricConfig]):
        """
        This class is responsible for building prompts used in model evaluations,
        incorporating task-specific prompts and metric descriptions.

        Args:
            task_type (TaskTypes): Type of the task for which prompts are being built.
            metric_configs (List[MetricConfig]): List of metric configurations.

        """
        self.task_type = task_type
        self.metric_configs = metric_configs

    @property
    def base_prompt(self):
        """
        Generates a base prompt that includes metric descriptions and
        placeholders for metric values.

        Returns:
            str: The base prompt string.

        """
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
        """
        Combines the base prompt, task-specific prompt, and replaces
        placeholders with specified values.

        Args:
            task_prompt (str): Task-specific prompt.
            replacement_map (Dict[str, str]): Dictionary of placeholders and their values.

        Returns:
            str: The complete prompt.
        """

        prompt = self.base_prompt
        prompt += task_prompt
        for replace_key, replace_val in replacement_map.items():
            prompt = prompt.replace(replace_key, replace_val)

        return prompt
