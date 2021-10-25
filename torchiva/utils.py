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

from typing import Dict, Optional

from torch import nn


def module_from_config(name: str, kwargs: Optional[Dict] = None, **other):
    """
    Get a model by its name

    Parameters
    ----------
    name: str
        Name of the model class
    kwargs: dict
        A dict containing all the arguments to the model
    """

    if kwargs is None:
        kwargs = {}

    parts = name.split(".")
    obj = None

    if len(parts) == 1:
        raise ValueError("Can't find object without module name")

    else:
        module = __import__(".".join(parts[:-1]), fromlist=(parts[-1],))
        if hasattr(module, parts[-1]):
            obj = getattr(module, parts[-1])

    if obj is not None:
        return obj(**kwargs)
    else:
        raise ValueError(f"The model {name} could not be found")
