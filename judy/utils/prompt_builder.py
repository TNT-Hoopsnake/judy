from typing import Dict, List, Optional
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
        Generates a base prompt.

        Returns:
            str: The base prompt string.

        """
        return BASE_PROMPT

    def update_metrics(self, prompt: str, metrics_filter: Optional[List[str]] = None):
        """
        Updates a prompt to include metric descriptions and
        placeholders for metric values.

        Returns:
            str: The updated prompt string.

        """
        metric_descriptions = []
        metric_formats = []
        metrics = []
        if metrics_filter:
            metrics = [
                metric
                for metric in self.metric_configs
                if metric.name in metrics_filter
            ]
        else:
            metrics = self.metric_configs
        if not metrics:
            raise ValueError(
                f"No metrics defined for prompt: {prompt}. Cannot continue with evaluation"
            )
        for metric in metrics:
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
        self,
        task_prompt: str,
        replacement_map: Dict[str, str],
        metrics_filter: Optional[List[str]] = None,
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
        prompt = self.update_metrics(prompt, metrics_filter)
        prompt += task_prompt
        for replace_key, replace_val in replacement_map.items():
            prompt = prompt.replace(replace_key, replace_val)

        return prompt
