from sqlitedict import SqliteDict
import hashlib
from gpt_eval.config.constants import USER_CACHE_DIR
from gpt_eval.config import (
    EVAL_CONFIG_PATH,
    SYSTEM_CONFIG_PATH
)

class SqliteCache:

    def __init__(self):
        self.cache = SqliteDict(USER_CACHE_DIR / 'cache.db', autocommit=True)

    def calculate_content_hash(self, content):
        encoded_content = str(content).encode()
        return hashlib.sha256(encoded_content).hexdigest().encode()


    def calculate_file_hash(self, file_path):
        with open(file_path, 'rb') as file:
            content = file.read()
            return self.calculate_content_hash(content)


    def calculate_merkle_tree_hash(self, file_paths):
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
            tree_hashes = [hashlib.sha256(tree_hashes[i] + tree_hashes[i+1]).hexdigest() for i in range(0, len(tree_hashes), 2)]

        # The root hash is the collective hash
        return tree_hashes[0]

    def set(self, key, subkey, data):
        key += subkey
        self.cache[key] = data        

    def get(self, key, subkey):
        key += subkey
        return self.cache.get(key, None)

    def build_cache_key(self, ds_name, scenario_type):
        config_hash = self.calculate_merkle_tree_hash([EVAL_CONFIG_PATH, SYSTEM_CONFIG_PATH])
        cache_key = f"{config_hash}-{scenario_type}-{ds_name}"
        return cache_key
