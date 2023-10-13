from sqlitedict import SqliteDict
import hashlib
from gpt_eval.config import (
    EVAL_CONFIG_PATH,
    SYSTEM_CONFIG_PATH
)

def calculate_content_hash(content):
    encoded_content = str(content).encode()
    return hashlib.sha256(encoded_content).hexdigest().encode()


def calculate_file_hash(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
        return calculate_content_hash(content)


def calculate_merkle_tree_hash(file_paths):
    if not file_paths:
        return None

    # Calculate leaf hashes for all files
    leaf_hashes = [calculate_file_hash(file_path) for file_path in file_paths]

    # Ensure the number of leaves is even by duplicating the last one if needed
    if len(leaf_hashes) % 2 != 0:
        leaf_hashes.append(leaf_hashes[-1])

    # Build the Merkle Tree structure (balanced binary tree)
    tree_hashes = leaf_hashes
    while len(tree_hashes) > 1:
        tree_hashes = [hashlib.sha256(tree_hashes[i] + tree_hashes[i+1]).hexdigest() for i in range(0, len(tree_hashes), 2)]

    # The root hash is the collective hash
    return tree_hashes[0]


cache = SqliteDict('./cache.db', autocommit=True)

def set_cache(key, subkey, data):
    key += subkey
    cache[key] = data        

def get_cache(key, subkey):
    key += subkey
    return cache.get(key, None)

def build_cache_key(ds_name, scenario_type):
    config_hash = calculate_merkle_tree_hash([EVAL_CONFIG_PATH, SYSTEM_CONFIG_PATH])
    cache_key = f"{config_hash}-{scenario_type}-{ds_name}"
    return cache_key
