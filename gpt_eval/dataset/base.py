from typing import List, Any
from abc import ABC
from datasets import Dataset


class BaseFormatter(ABC):
    def __init__(self, dataset: Dataset, eval_idxs: List[int]):
        self.dataset = dataset
        self.eval_idxs = eval_idxs

    def format(self) -> Any:
        pass
