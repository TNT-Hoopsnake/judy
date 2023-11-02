import pathlib
import shutil

from gpt_eval.config.settings import DATASETS_DIR, USER_CACHE_DIR, USER_CONFIG_DIR


def setup_user_dir():
    # Setup config files
    if not USER_CONFIG_DIR.is_dir():
        USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for file in (pathlib.Path(__file__).parent.parent / "config/files/").iterdir():
        new_file_name = file.name.replace("example_", "")
        if not (USER_CONFIG_DIR / f"{new_file_name}").is_file():
            shutil.copyfile(file, USER_CONFIG_DIR / f"{new_file_name}")

    # Setup cache dir
    if not USER_CACHE_DIR.is_dir():
        USER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Setup datasets
    if not DATASETS_DIR.is_dir():
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
