import hashlib
from pathlib import Path
from typing import Any, List
from sqlitedict import SqliteDict
from gpt_eval.config import EVAL_CONFIG_PATH, METRIC_CONFIG_PATH
from gpt_eval.config.constants import USER_CACHE_DIR


class SqliteCache:
    def __init__(self):
        self.cache = SqliteDict(USER_CACHE_DIR / "cache.db", autocommit=True)

    def calculate_content_hash(self, content: bytes) -> str:
        encoded_content = str(content).encode()
        return hashlib.sha256(encoded_content).hexdigest()

    def calculate_file_hash(self, file_path: str | Path) -> str:
        with open(file_path, "rb") as file:
            content = file.read()
            return self.calculate_content_hash(content)

    def calculate_merkle_tree_hash(self, file_paths: List[str | Path]) -> str | None:
        if not file_paths:
            return None

        # Calculate leaf hashes for all files
        leaf_hashes = [self.calculate_file_hash(file_path) for file_path in file_paths]

        # Ensure the number of leaves is even by duplicating the last one if needed
        if len(leaf_hashes) % 2 != 0:
            leaf_hashes.append(leaf_hashes[-1])

        # Build the Merkle Tree structure (balanced binary tree)
        tree_hashes = leaf_hashes
        while len(tree_hashes) > 1:
            tree_hashes = [
                hashlib.sha256(
                    (tree_hashes[i] + tree_hashes[i + 1]).encode()
                ).hexdigest()
                for i in range(0, len(tree_hashes), 2)
            ]

        # The root hash is the collective hash
        return str(tree_hashes[0])

    def set(self, key: str, subkey: str, data: Any):
        key += subkey
        self.cache[key] = data

    def get(self, key: str, subkey: str):
        key += subkey
        return self.cache.get(key, None)

    def build_cache_key(self, ds_name: str, scenario_type: str):
        config_hash = self.calculate_merkle_tree_hash(
            [EVAL_CONFIG_PATH, METRIC_CONFIG_PATH]
        )
        cache_key = f"{config_hash}-{scenario_type}-{ds_name}"
        return cache_key
