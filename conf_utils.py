# Copyright 2021 Robin Scheibler
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

