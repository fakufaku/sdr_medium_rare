import json
from pathlib import Path
from typing import Union, List, Dict
import yaml

def read_config(filename: Union[str, Path]) -> Union[List, Dict]:
    filename = Path(filename)
    with open(filename, "r") as f:
        if filename.suffix == ".json":
            config = json.load(f)
        elif filename.suffix == ".yml" or filename.suffix == ".yaml":
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError(f"Config filetype {filename.suffix} not supported")

    return config

def write_config(content: Union[Dict, List], filename: Union[str, Path]):

    filename = Path(filename)
    with open(filename, "w") as f:
        if filename.suffix == ".json":
            config = json.dump(content, f, indent=4)
        elif filename.suffix == ".yml" or filename.suffix == ".yaml":
            config = yaml.dump(content, f)
        else:
            raise ValueError(f"Config filetype {filename.suffix} not supported")

