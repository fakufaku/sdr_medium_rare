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

import abc
import math
from enum import Enum
from typing import Optional, Union

import torch as pt


class Window(Enum):
    CUSTOM = None
    BARTLETT = "bartlett"
    BLACKMAN = "blackman"
    HAMMING = "hamming"
    HANN = "hann"


window_types = [s for s in Window._value2member_map_ if s is not None]


class SourceModelBase(pt.nn.Module):
    """
    An abstract class to represent source models

    Parameters
    ----------
    X: numpy.ndarray or torch.Tensor, shape (..., n_frequencies, n_frames)
        STFT representation of the signal

    Returns
    -------
    P: numpy.ndarray or torch.Tensor, shape (..., n_frequencies, n_frames)
        The inverse of the source power estimate
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        """
        The reset method is intended for models that have some internal state
        that should be reset for every new signal.

        By default, it does nothing and should be overloaded when needed by
        a subclass.
        """
        pass


class STFTBase(abc.ABC):
    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[Union[Window, str]] = None,
    ):
        self._n_fft = n_fft
        self._n_freq = n_fft // 2 + 1
        self._hop_length = hop_length

        if window is None:
            self._window_type = Window.HAMMING
        else:
            self._window_type = Window(window)

        # defer window creation to derived class
        self._window = None

    def _get_n_frames(self, n_samples):
        n_hop = math.floor(n_samples / self.hop_length)
        return n_hop + 1

    @property
    def n_fft(self):
        return self._n_fft

    @property
    def hop_length(self):
        return self._hop_length

    @property
    def n_freq(self):
        return self._n_freq

    @property
    def window(self):
        return self._window

    @property
    def window_type(self):
        return self._window_type

    @property
    def window_name(self):
        return self._window_type.value

    @abc.abstractmethod
    def _make_window(self, dtype):
        pass

    @abc.abstractmethod
    def _forward(self, x):
        pass

    @abc.abstractmethod
    def _backward(self, x):
        pass

    def __call__(self, x):
        self._make_window(x)
        return self._forward(x)

    def inv(self, x):
        self._make_window(x)
        return self._backward(x)
