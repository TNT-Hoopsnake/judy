import pathlib
import shutil

def setup_user_dir():
    # Setup the base directory
    base_dir = pathlib.Path.home() / ".gpt-eval"
    if not base_dir.is_dir():
        base_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup config files
    config_dir = base_dir / "config"
    if not config_dir.is_dir():
        config_dir.mkdir(parents=True, exist_ok=True)
    for file in (pathlib.Path(__file__).parent.parent/ "config/files/").iterdir():
        new_file_name = file.name.replace("example_", "")
        if not (config_dir / f"{new_file_name}").is_file():
            shutil.copyfile(file, config_dir / f"{new_file_name}")

    # Setup cache dir
    cache_dir = base_dir / "cache"
    if not cache_dir.is_dir():
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Setup datasets
    data_dir = base_dir / "datasets"
    if not data_dir.is_dir():
        data_dir.mkdir(parents=True, exist_ok=True)
