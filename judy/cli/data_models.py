import asyncio
from typing import List
from pydantic import BaseModel, ConfigDict


class TaskPipelineStage(BaseModel):
    """A stage in an asyncio pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    tasks: List[asyncio.Task]
    input_queue: asyncio.Queue
    output_queue: asyncio.Queue
    name: str
    description: str

    def add_task(self, task: asyncio.Task):
        """Add a task to the stage."""
        self.tasks.append(task)


class TaskPipeline(BaseModel):
    """An asyncio pipeline for processing tasks in stages."""

    _stages: List[TaskPipelineStage]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add_stage(self, name: str, description: str):
        """Add a stage to the pipeline."""
        if self._stages:
            self._stages = [
                TaskPipelineStage(
                    tasks=[],
                    input_queue=asyncio.Queue(),
                    output_queue=asyncio.Queue(),
                    name=name,
                    description=description,
                )
            ]
        else:
            self._stages.append(
                TaskPipelineStage(
                    tasks=[],
                    input_queue=self._stages[-1].output_queue,
                    output_queue=asyncio.Queue(),
                    name=name,
                    description=description,
                )
            )

    def next(self):
        """Get the next stage in the pipeline."""
        return self._stages.pop(0)
