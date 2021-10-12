from typing import Optional, Tuple
import argparse
import json
import multiprocessing
import os
import random
import re
import time
from pathlib import Path

from ci_sdr.pt.sdr import ci_sdr_loss_hungarian
import numpy as np
import torch
import yaml
from scipy.io import wavfile

from conf_utils import write_config
import fast_bss_eval
from mixer import mixer

n_repeat = 10
batch_size = 10
cgd_n_iter = 10
n_channels = [2, 3, 4, 5, 6, 7, 8]
filter_lengths = [512, 1024]
signal_lengths = [5.0, 20.0]
target_sdr = 15
target_sir = 300
device = 0


def get_mix(
    n_batch,
    n_chan,
    length_s,
    target_sdr,
    target_sir,
    fs=16000,
    filter_length=512,
    device="cpu",
):
    n_samples = int(length_s * fs)
    ref = torch.zeros((n_batch, n_chan, n_samples)).normal_().to(device)
    mix = mixer(ref, target_sdr, target_sir, filter_len=filter_length)

    return ref, mix


class TimeLogger:
    def __init__(self):
        self.timestamps = {"create": time.perf_counter()}

    def log(self, label):
        self.timestamps[label] = time.perf_counter()

    def delta(self, label1, label2):
        return self.timestamps[label2] - self.timestamps[label1]


def init_load_dummy(device="cpu"):

    ref = torch.zeros((2, 16000 * 3)).normal_().to(device)
    est = torch.zeros((2, 16000 * 3)).normal_().to(device)

    fast_bss_eval.ci_sdr(ref, est, use_cg_iter=cgd_n_iter)
    fast_bss_eval.ci_sdr(ref, est)
    ci_sdr_loss_hungarian(est, ref)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test accuracy of CGD vs Solve for BSS Eval"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="./experiments_sdr/output",
        help="Path to output folder",
    )
    args = parser.parse_args()

    output_filename = "runtime_accuracy.json"

    init_load_dummy(device)

    results = []

    for sig_len in signal_lengths:
        for fil_len in filter_lengths:
            for n_chan in n_channels:

                t_ci_sdr = 0.0
                t_prop_solve = 0.0
                t_prop_cgd = 0.0
                e_prop_solve = 0.0
                e_prop_cgd = 0.0

                for epoch in range(n_repeat):

                    ref, mix = get_mix(
                        batch_size,
                        n_chan,
                        sig_len,
                        target_sdr,
                        target_sir,
                        filter_length=100,
                        device=device,
                    )

                    try:
                        t = time.perf_counter()
                        sdr_cisdr = -ci_sdr_loss_hungarian(
                            mix, ref, filter_length=fil_len
                        )
                        t_ci_sdr += time.perf_counter() - t
                        success_ci_sdr = True
                    except RuntimeError:
                        t_ci_sdr = -1
                        success_ci_sdr = False

                    try:
                        t = time.perf_counter()
                        sdr_prop_solve = fast_bss_eval.ci_sdr(
                            ref, mix, filter_length=fil_len
                        )
                        t_prop_solve += time.perf_counter() - t
                        e_prop_solve += torch.mean((sdr_cisdr - sdr_prop_solve).abs())
                        success_prop_solve = True
                    except RuntimeError:
                        t_prop_solve = -1
                        e_prop_solve = -1
                        success_prop_solve = False

                    try:
                        t = time.perf_counter()
                        sdr_prop_cgd = fast_bss_eval.ci_sdr(
                            ref, mix, filter_length=fil_len, use_cg_iter=cgd_n_iter
                        )
                        t_prop_cgd += time.perf_counter() - t
                        e_prop_cgd += torch.mean((sdr_cisdr - sdr_prop_cgd).abs())
                        success_prop_cgd = True
                    except RuntimeError:
                        t_prop_cgd = -1
                        e_prop_cgd = -1
                        success_prop_cgd = False

                t_prop_solve /= n_repeat
                t_prop_cgd /= n_repeat
                t_ci_sdr /= n_repeat
                e_prop_solve /= n_repeat
                e_prop_cgd /= n_repeat

                print(f"{n_chan}ch sig_len={sig_len} fil_len={fil_len}")
                print(f"  - Time ci_sdr: {t_ci_sdr:.6f}")
                print(f"  - Time solve : {t_prop_solve:.6f}")
                print(f"  - Time cgd   : {t_prop_cgd:.6f}")
                print(f"  - Error solve: {e_prop_solve:.6f}")
                print(f"  - Error cgd  : {e_prop_cgd:.6f}")
                print()

                results += [
                    {
                        "signal_length": sig_len,
                        "filter_length": fil_len,
                        "channels": n_chan,
                        "algo": "cisdr",
                        "success": success_ci_sdr,
                        "runtime": t_ci_sdr,
                        "speedup": 1.0,
                        "error": 0.0,
                    },
                    {
                        "signal_length": sig_len,
                        "filter_length": fil_len,
                        "channels": n_chan,
                        "algo": "solve",
                        "success": success_prop_solve,
                        "runtime": t_prop_solve,
                        "speedup": t_ci_sdr / t_prop_solve,
                        "error": float(e_prop_solve),
                    },
                    {
                        "signal_length": sig_len,
                        "filter_length": fil_len,
                        "channels": n_chan,
                        "algo": "cgd",
                        "success": success_prop_cgd,
                        "runtime": t_prop_cgd,
                        "speedup": t_ci_sdr / t_prop_cgd,
                        "error": float(e_prop_cgd),
                    },
                ]

    write_config(results, args.output / "runtime_vs_ci_sdr.json")
