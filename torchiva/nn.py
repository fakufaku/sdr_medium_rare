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

import enum
from typing import List, Optional, Dict, Union, Callable

import torch as pt
from torch import nn

from .auxiva_iss import auxiva_iss
from .overiva import overiva
from .models import LaplaceModel
from .scaling import minimum_distortion, minimum_distortion_l2_phase, projection_back
from .stft import STFT
from .utils import module_from_config


class SepAlgo(enum.Enum):
    AUXIVA_ISS = "auxiva-iss"
    AUXIVA_IP2 = "auxiva-ip2"
    OVERIVA_IP = "overiva-ip"


class Separator(nn.Module):
    def __init__(
        self,
        n_fft: int,
        n_src: Optional[int] = None,
        algo: Optional[SepAlgo] = SepAlgo.AUXIVA_ISS,
        hop_length: Optional[int] = None,
        window: Optional[int] = None,
        source_model: Optional[Callable] = None,
        n_iter: Optional[int] = 10,
        ref_mic: Optional[int] = 0,
        mdp_p: Optional[float] = None,
        mdp_q: Optional[float] = None,
        proj_back: Optional[bool] = False,
        mdp_phase: Optional[bool] = False,
        mdp_model: Optional[nn.Module] = None,
        postfilter_model: Optional[nn.Module] = None,
        algo_kwargs: Optional[Dict] = None,
        checkpoints: Optional[List[int]] = None,
    ):

        super().__init__()

        if source_model is None:
            self.source_model = LaplaceModel()
        else:
            if isinstance(source_model, dict):
                self.source_model = module_from_config(**source_model)
            else:
                self.source_model = source_model

        self.n_src = n_src
        self.algo = algo
        if algo_kwargs is None:
            self.algo_kwargs = {}
        else:
            self.algo_kwargs = algo_kwargs

        # other attributes
        self.ref_mic = ref_mic
        self.n_iter = n_iter
        self.mdp_p = mdp_p
        self.mdp_q = mdp_q
        self.proj_back = proj_back
        self.mdp_phase = mdp_phase

        if isinstance(mdp_model, dict):
            self.mdp_model = module_from_config(**mdp_model)
        else:
            self.mdp_model = mdp_model

        # we can also optionnaly use a post-filter masking model
        if isinstance(postfilter_model, dict):
            self.postfilter_model = module_from_config(**postfilter_model)
        else:
            self.postfilter_model = postfilter_model

        # the stft engine
        self.stft = STFT(n_fft, hop_length=hop_length, window=window)

        # in some cases, we want to instrument
        self.checkpoints = checkpoints
        self.saved_checkpoints = None

    @property
    def algo(self):
        return self._algo.value

    @algo.setter
    def algo(self, val):
        self._algo = SepAlgo(val)

    def scale(self, Y, X):

        if self.proj_back:
            Y = projection_back(Y, X[..., self.ref_mic, :, :])
        elif self.mdp_phase:
            Y = minimum_distortion_l2_phase(Y, X[..., self.ref_mic, :, :],)
        elif self.mdp_model is not None or self.mdp_p is not None:
            Y = minimum_distortion(
                Y,
                X[..., self.ref_mic, :, :],
                p=self.mdp_p,
                q=self.mdp_q,
                model=self.mdp_model,
                max_iter=1,
            )
        else:
            pass

        return Y

    def forward(self, x, n_iter=None, reset=True):
        if n_iter is None:
            n_iter = self.n_iter

        if hasattr(self.source_model, "reset") and reset:
            # this is required for models with internal state, such a ILRMA
            self.source_model.reset()

        n_chan, n_samples = x.shape[-2:]

        if self.n_src is None:
            n_src = n_chan
        else:
            n_src = self.n_src

        assert n_chan >= n_src, (
            "The number of channel should be larger or equal to "
            "the number of sources to separate."
        )

        # prepare a list for checkpoints just in case
        saved_checkpoints = []

        # STFT
        X = self.stft(x)  # copy for back projection (numpy/torch compatible)

        # Separation
        if self._algo == SepAlgo.AUXIVA_ISS:
            Y = auxiva_iss(
                X,
                n_iter=n_iter,
                model=self.source_model,
                two_chan_ip2=False,
                checkpoints_iter=self.checkpoints,
                checkpoints_list=saved_checkpoints,
                **self.algo_kwargs,
            )

        elif self._algo == SepAlgo.AUXIVA_IP2:
            Y = auxiva_iss(
                X,
                n_iter=n_iter,
                model=self.source_model,
                two_chan_ip2=True,
                checkpoints_iter=self.checkpoints,
                checkpoints_list=saved_checkpoints,
                **self.algo_kwargs,
            )

        elif self._algo == SepAlgo.OVERIVA_IP:
            Y = overiva(
                X,
                n_src=self.n_src,
                n_iter=n_iter,
                model=self.source_model,
                checkpoints_iter=self.checkpoints,
                checkpoints_list=saved_checkpoints,
                **self.algo_kwargs,
            )

        else:
            raise NotImplementedError

        # Solve scale ambiguity
        Y = self.scale(Y, X)

        # iSTFT
        y = self.stft.inv(Y)  # (n_samples, n_channels)

        if self.checkpoints is not None:
            self.saved_checkpoints = {
                "iter": self.checkpoints[: len(saved_checkpoints)],
                "checkpoints": saved_checkpoints,
                "ref": X,
            }

        if y.shape[-1] < x.shape[-1]:
            y = pt.cat(
                (y, y.new_zeros(y.shape[:-1] + (x.shape[-1] - y.shape[-1],))), dim=-1
            )
        elif y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]

        return y

    def process_checkpoints(self):
        """
        We adjust scale and inv-STFT all the saved spectrograms
        """
        if self.saved_checkpoints is None:
            return

        # concatenate and broad cast to the same shape
        X = self.saved_checkpoints["ref"]
        Ys = pt.cat(
            [Y[None, ...] for Y in self.saved_checkpoints["checkpoints"]], dim=0
        )
        X, Ys = pt.broadcast_tensors(X[None, ...], Ys)

        # adjust scale
        Ys = self.scale(Ys, X)

        # Post-filtering
        Ys = self.postfilter(Ys)

        # transform to time-domain
        ys = self.stft.inv(Ys)

        ret = {"iter": self.saved_checkpoints["iter"], "signals": ys}
        self.saved_checkpoints = None

        return ret
