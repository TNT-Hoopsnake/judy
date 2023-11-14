import hashlib
import os
from pathlib import Path
from typing import Any, List
from sqlitedict import SqliteDict
from judy.config.settings import USER_CACHE_DIR
from judy.config.logging import logger as log


class SqliteCache:
    def __init__(self, config_paths: List[str | Path], clear_cache: bool):
        """
        Initialize the SqliteCache instance.

        Args:
            config_paths (List[str | Path]): List of paths to configuration files.
            clear_cache (bool): Flag to clear the cache.

        The cache utilizes a SHA-256 Merkle Tree hash of the content of configuration files
        provided in `config_paths` to create a new table in the cache for the current state
        of these files. If any of the configuration files defined in `config_paths` are changed,
        the hash will be updated, and a new cache table will be created.
        """
        cache_path = USER_CACHE_DIR / "cache.db"
        if clear_cache and Path(cache_path).is_file():
            log.info("Destroyed cache at path: %s", cache_path)
            os.remove(cache_path)

        root_key = self.build_root_key(config_paths)
        self.cache = SqliteDict(cache_path, autocommit=True, tablename=root_key)

        log.info("Loaded cache from file: %s", cache_path)

    def calculate_content_hash(self, content: Any) -> str:
        """
        Calculate the SHA-256 hash of the provided content.

        Args:
            content (Any): The content to hash.

        Returns:
            str: The SHA-256 hash of the content.
        """
        encoded_content = str(content).encode()
        return hashlib.sha256(encoded_content).hexdigest()

    def calculate_file_hash(self, file_path: str | Path) -> str:
        """
        Calculate the SHA-256 hash of the content of a file.

        Args:
            file_path (str | Path): Path to the file.

        Returns:
            str: The SHA-256 hash of the file content.
        """
        with open(file_path, "rb") as file:
            content = file.read()
            return self.calculate_content_hash(content)

    def calculate_merkle_tree_hash(self, config_paths) -> str | None:
        """
        Calculate the Merkle Tree hash based on the leaf hashes of files.

        Args:
            config_paths: List of paths to configuration files.

        Returns:
            str | None: The calculated Merkle Tree hash or None if the list of config paths is empty.
        """
        # Calculate leaf hashes for all files
        leaf_hashes = [
            self.calculate_file_hash(file_path) for file_path in config_paths
        ]

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
        log.debug("Added to cache using key: %s", key)
        self.cache[key] = data

    def get(self, key: str, subkey: str):
        key += subkey
        log.debug("Retrieved from cache using key: %s", key)
        return self.cache.get(key, None)

    def build_cache_key(self, ds_name: str, task_type: str):
        """
        Build a cache key based on dataset name and task type.

        Args:
            ds_name (str): Dataset name.
            task_type (str): Task type.

        Returns:
            str: The built cache key.
        """
        cache_key = f"{task_type}-{ds_name}"
        log.debug("Built cache key: %s", cache_key)
        return cache_key

    def build_root_key(self, config_paths):
        """
        Build the root key for the cache based on the Merkle Tree hash of configuration files.

        Args:
            config_paths: List of paths to configuration files.
        """
        root_key = self.calculate_merkle_tree_hash(config_paths)
        return root_key
