from typing import Optional, Tuple
import argparse
import json
import multiprocessing
import os
import random
import re
import time
from pathlib import Path

import ci_sdr
import numpy as np
import torch
import yaml
from mir_eval.separation import bss_eval_sources as mir_eval_bss_eval_sources
from bsseval.metrics import bss_eval_sources as sigsep_bss_eval_sources
from scipy.io import wavfile

import torchiva
from datasets import WSJ1SpatialDataset

from conf_utils import write_config

from fast_bss_eval.torch.cgd import (
    toeplitz_conjugate_gradient,
    block_toeplitz_conjugate_gradient,
)
from fast_bss_eval.torch.linalg import toeplitz, block_toeplitz
from fast_bss_eval.torch.metrics import (
    compute_stats_2,
    _base_metrics_bss_eval,
    bss_eval_sources,
    ci_sdr,
)
from fast_bss_eval.torch.helpers import _solve_permutation, _normalize
import fast_bss_eval

algo_config = {
    "name": "torchiva.nn.Separator",
    "kwargs": {
        "n_fft": 4096,
        "algo": "overiva-ip",
        "n_iter": 40,
        "ref_mic": 0,
        "mdp_p": 2.0,
        "source_model": {"name": "torchiva.models.GaussModel", "kwargs": {}},
    },
}

cgd_n_iter = [1, 2, 3, 5, 7, 10, 15, 20, 30]
filter_lengths = [512]


class TimeLogger:
    def __init__(self):
        self.timestamps = {"create": time.perf_counter()}

    def log(self, label):
        self.timestamps[label] = time.perf_counter()

    def delta(self, label1, label2):
        return self.timestamps[label2] - self.timestamps[label1]


def metrics_from_solutions(sdr_sol, sir_sol, xcorr_sdr, xcorr_sir):

    # pairwise coherence
    coh_sdr = torch.einsum(
        "...lc,...lc->...c", xcorr_sdr, sdr_sol
    )  # (..., n_ref, n_est)

    coh_sar = torch.einsum("...lc,...lc->...c", xcorr_sir, sir_sol)

    coh_sdr, coh_sar = torch.broadcast_tensors(coh_sdr, coh_sar[..., None, :])

    coh_sdr = torch.clamp(coh_sdr, min=1e-7, max=1.0 - 1e-7)
    coh_sar = torch.clamp(coh_sar, min=1e-7, max=1.0 - 1e-7)

    neg_sdr, neg_sir, neg_sar = _base_metrics_bss_eval(coh_sdr, coh_sar, clamp_db=150)

    return neg_sdr, neg_sir, neg_sar


def instrumented_mir_eval_bss_eval_sources(
    ref: torch.Tensor, est: torch.Tensor,
):

    ref = ref.cpu().to(torch.float64).numpy()
    est = est.cpu().to(torch.float64).numpy()

    t = time.perf_counter()
    outputs = mir_eval_bss_eval_sources(ref, est)
    mir_eval_runtime = time.perf_counter() - t

    sdr, sir, sar, perm = [torch.from_numpy(o) for o in outputs]
    mir_eval_metrics = {
        "sdr": sdr.tolist(),
        "sir": sir.tolist(),
        "sar": sar.tolist(),
        "perm": perm.tolist(),
    }

    t = time.perf_counter()
    outputs = sigsep_bss_eval_sources(ref, est)
    sigsep_runtime = time.perf_counter() - t

    sdr, sir, sar, perm = [torch.from_numpy(o) for o in outputs]
    sigsep_metrics = {
        "sdr": sdr[:, 0].tolist(),
        "sir": sir[:, 0].tolist(),
        "sar": sar[:, 0].tolist(),
        "perm": perm[:, 0].tolist(),
    }

    return mir_eval_metrics, mir_eval_runtime, sigsep_metrics, sigsep_runtime


def instrumented_fast_bss_eval_sources(
    ref: torch.Tensor,
    est: torch.Tensor,
    use_cg_iter=None,
    use_fp64=False,
    use_numpy=False,
):

    if use_fp64:
        ref = ref.to(torch.float64)
        est = est.to(torch.float64)

    if use_numpy:
        ref = ref.cpu().numpy()
        est = est.cpu().numpy()

    t = time.perf_counter()
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(
        ref, est, use_cg_iter=use_cg_iter
    )
    runtime = time.perf_counter() - t

    metrics = {
        "sdr": sdr.tolist(),
        "sir": sir.tolist(),
        "sar": sar.tolist(),
        "perm": perm.tolist(),
    }

    return metrics, runtime


def instrumented_bss_eval_sources(
    ref: torch.Tensor,
    est: torch.Tensor,
    filter_length: Optional[int] = 512,
    use_cg_iter: Optional[int] = [10],
    pairwise: Optional[bool] = True,
    load_diag: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    time_logger = TimeLogger()
    time_logger.log(0)

    # normalize along time-axis
    est = _normalize(est, dim=-1)
    ref = _normalize(ref, dim=-1)

    time_logger.log(1)

    # compute auto-correlation and cross-correlation
    # acf.shape = (..., 2 * filter_length, n_ref, n_est)
    # xcorr.shape = (..., filter_length, n_ref, n_est)
    acf, xcorr = compute_stats_2(ref, est, length=filter_length)

    time_logger.log(2)

    diag_indices = list(range(ref.shape[-2]))

    if load_diag is not None:
        # the diagonal factor of the Toeplitz matrix is the first
        # coefficient of the acf
        acf[..., 0, diag_indices, diag_indices] += load_diag

    # solve for the SDR
    acf_sdr = acf[..., diag_indices, diag_indices]
    acf_sdr = acf_sdr.transpose(-2, -1)
    acf_sdr = acf_sdr[..., :filter_length]

    xcorr_sdr = xcorr.transpose(-3, -2)

    time_logger.log(3)

    # solve for the optimal filter
    # regular matrix solver
    R_mat = toeplitz(acf_sdr)
    sol_sdr_solve = torch.linalg.solve(R_mat, xcorr_sdr)

    time_logger.log(4)

    # use preconditioned conjugate gradient
    sol_sdr_cgd = {}
    sdr_cgd_runtimes = []
    for n_iter in use_cg_iter:
        t0 = time.perf_counter()
        sol_sdr_cgd[n_iter] = toeplitz_conjugate_gradient(
            acf_sdr, xcorr_sdr, n_iter=n_iter
        )
        sdr_cgd_runtimes.append(time.perf_counter() - t0)

    time_logger.log(5)

    # solve the coefficients for the SIR
    xcorr = xcorr.reshape(xcorr.shape[:-3] + (-1,) + xcorr.shape[-1:])

    time_logger.log(6)

    R_mat = block_toeplitz(acf)
    sol_sir_solve = torch.linalg.solve(R_mat, xcorr)

    time_logger.log(8)

    sol_sir_cgd = {}
    sir_cgd_runtimes = []
    for n_iter in use_cg_iter:
        t0 = time.perf_counter()
        x0 = sol_sdr_cgd[n_iter].transpose(-3, -2)
        x0 = x0.reshape(x0.shape[:-3] + (-1,) + x0.shape[-1:])
        sol_sir_cgd[n_iter] = block_toeplitz_conjugate_gradient(
            acf, xcorr, n_iter=n_iter, x=x0
        )
        sir_cgd_runtimes.append(time.perf_counter() - t0)

    time_logger.log(9)

    # the values obtained from the solve function
    neg_sdr, neg_sir, neg_sar = metrics_from_solutions(
        sol_sdr_solve, sol_sir_solve, xcorr_sdr, xcorr
    )

    time_logger.log(10)

    # now compute for CGD
    metrics_cgd = {}
    errors_cgd = {}
    for n_iter in use_cg_iter:

        dsdr_sol = torch.mean(torch.abs(sol_sdr_solve - sol_sdr_cgd[n_iter]))
        dsir_sol = torch.mean(torch.abs(sol_sir_solve - sol_sir_cgd[n_iter]))

        nsdr, nsir, nsar = metrics_from_solutions(
            sol_sdr_cgd[n_iter], sol_sir_cgd[n_iter], xcorr_sdr, xcorr
        )

        # for CGD vs Solve, we compute diff before permutation
        errors_cgd[n_iter] = {
            "d_toep_cgd": dsdr_sol.tolist(),
            "d_blk_toep_cgd": dsir_sol.tolist(),
            "dsdr": (neg_sdr - nsdr).abs().mean().tolist(),
            "dsir": (neg_sir - nsir).abs().mean().tolist(),
            "dsar": (neg_sar - nsar).abs().mean().tolist(),
        }

        nsir, nsdr, nsar, perm = _solve_permutation(nsir, nsdr, nsar, return_perm=True)
        metrics_cgd[n_iter] = {
            "sdr": (-nsdr).tolist(),
            "sir": (-nsir).tolist(),
            "sar": (-nsar).tolist(),
            "perm": perm.tolist(),
        }

    time_logger.log(11)

    neg_sir, neg_sdr, neg_sar, perm = _solve_permutation(
        neg_sir, neg_sdr, neg_sar, return_perm=True
    )

    time_logger.log(12)

    # runtimes
    t_total = time_logger.delta(0, 12)
    t_acr_xcorr = time_logger.delta(1, 2)
    t_sdr_solve_direct = time_logger.delta(3, 4)
    t_sdr_solve_cgd = time_logger.delta(4, 5)
    t_sir_solve_direct = time_logger.delta(6, 8)
    t_sir_solve_cgd = time_logger.delta(8, 9)
    t_metrics = time_logger.delta(9, 10)
    t_metrics_cgd = time_logger.delta(10, 11)
    t_permute = time_logger.delta(11, 12)
    # others
    # t_block_diag_extract = time_logger.delta(2, 3)
    # t_xcorr_reshape = time_logger.delta(5, 6)
    # t_norm = time_logger.delta(0, 1)

    runtimes = {
        "total": t_total,
        "acf_xcorr": t_acr_xcorr,
        "toeplitz_solve": t_sdr_solve_direct,
        "toeplitz_cgd_lst": sdr_cgd_runtimes,
        "block_toeplitz_solve": t_sir_solve_direct,
        "block_toeplitz_cgd_lst": sir_cgd_runtimes,
        "coh_to_metrics": t_metrics,
        "permutation": t_permute,
        "other": (
            t_total
            - (t_acr_xcorr + t_sdr_solve_direct + t_sdr_solve_cgd + t_sir_solve_direct)
            - (t_sir_solve_cgd + t_metrics + t_permute + t_metrics_cgd)
        ),
    }

    metrics_solve = {
        "sdr": (-neg_sdr).tolist(),
        "sir": (-neg_sir).tolist(),
        "sar": (-neg_sar).tolist(),
        "perm": perm.tolist(),
    }

    return metrics_solve, metrics_cgd, errors_cgd, runtimes


def init_load_dummy():

    ref = torch.zeros((2, 16000 * 3)).normal_()
    est = torch.zeros((2, 16000 * 3)).normal_()

    instrumented_bss_eval_sources(ref, est, use_cg_iter=cgd_n_iter)
    instrumented_mir_eval_bss_eval_sources(ref, est)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test accuracy of CGD vs Solve for BSS Eval"
    )
    parser.add_argument("dataset_path", type=Path, help="Path to dataset")
    parser.add_argument(
        "--limit", type=int, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--with-mir-eval",
        action="store_true",
        help="Include mir_eval in the evaluation",
    )
    parser.add_argument(
        "--with-multithread",
        action="store_true",
        help="Include mir_eval in the evaluation",
    )
    parser.add_argument(
        "output",
        type=Path,
        default="./experiment_sdr/output",
        help="Path to output folder",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    output_filename = "runtime_accuracy.json"

    if args.with_mir_eval:
        use_gpu = False

        if args.with_multithread:
            output_filename = "runtime_accuracy_vs_mir_eval_multithread.json"

        else:
            output_filename = "runtime_accuracy_vs_mir_eval.json"

            # single threaded computations
            torch.set_num_threads(1)

    else:
        output_filename = "runtime_accuracy_with_gpu.json"
        use_gpu = True

    # call this too ensure all the underlying libraries are loaded already
    init_load_dummy()

    # Create the separation function
    separator = torchiva.utils.module_from_config(**algo_config)

    data = []

    for channels in [2, 3, 4]:

        # get the data
        dataset = WSJ1SpatialDataset(
            args.dataset_path / f"wsj1_{channels}_mix_m{channels}/eval92",
            ref_mic=0,
            ref_is_reverb=True,
            remove_mean=True,
        )

        for idx, (mix, ref) in enumerate(dataset):

            if args.limit is not None and idx >= args.limit:
                break

            if use_gpu:
                mix = mix.to(0)
                ref = ref.to(0)

            for with_bss in [False, True]:

                if with_bss:
                    mix.to(0)  # always GPU for separation
                    est = separator(mix)

                    if not use_gpu:
                        est = est.to("cpu")

                    data_id = idx + len(dataset)

                else:
                    est = mix
                    data_id = idx

                metrics_solve_32, runtime_solve_32 = instrumented_fast_bss_eval_sources(
                    ref, est, use_cg_iter=None, use_fp64=False, use_numpy=False
                )
                metrics_solve_64, runtime_solve_64 = instrumented_fast_bss_eval_sources(
                    ref, est, use_cg_iter=None, use_fp64=True, use_numpy=False
                )

                metrics_cgd_32, runtime_cgd_32 = instrumented_fast_bss_eval_sources(
                    ref, est, use_cg_iter=10, use_fp64=False, use_numpy=False
                )
                metrics_cgd_64, runtime_cgd_64 = instrumented_fast_bss_eval_sources(
                    ref, est, use_cg_iter=10, use_fp64=True, use_numpy=False
                )

                data.append(
                    {
                        "data_id": data_id,
                        "channels": channels,
                        "metrics_prop_solve_32": metrics_solve_32,
                        "runtime_prop_solve_32": runtime_solve_32,
                        "metrics_prop_solve_64": metrics_solve_64,
                        "runtime_prop_solve_64": runtime_solve_64,
                        "metrics_prop_cgd_32": metrics_cgd_32,
                        "runtime_prop_cgd_32": runtime_cgd_32,
                        "metrics_prop_cgd_64": metrics_cgd_64,
                        "runtime_prop_cgd_64": runtime_cgd_64,
                    }
                )

                if not use_gpu:
                    # numpy
                    npy32_met_solve, npy32_rt_solve = instrumented_fast_bss_eval_sources(
                        ref, est, use_cg_iter=None, use_fp64=False, use_numpy=True
                    )
                    npy64_met_solve, npy64_rt_solve = instrumented_fast_bss_eval_sources(
                        ref, est, use_cg_iter=None, use_fp64=True, use_numpy=True
                    )

                    npy32_met_cgd, npy32_rt_cgd = instrumented_fast_bss_eval_sources(
                        ref, est, use_cg_iter=10, use_fp64=False, use_numpy=True
                    )
                    npy64_met_cgd, npy64_rt_cgd = instrumented_fast_bss_eval_sources(
                        ref, est, use_cg_iter=10, use_fp64=True, use_numpy=True
                    )


                    data[-1].update(
                        {
                            "metrics_prop_npy_solve_32": npy32_met_solve,
                            "runtime_prop_npy_solve_32": npy32_rt_solve,
                            "metrics_prop_npy_solve_64": npy64_met_solve,
                            "runtime_prop_npy_solve_64": npy64_rt_solve,
                            "metrics_prop_npy_cgd_32": npy32_met_cgd,
                            "runtime_prop_npy_cgd_32": npy32_rt_cgd,
                            "metrics_prop_npy_cgd_64": npy64_met_cgd,
                            "runtime_prop_npy_cgd_64": npy64_rt_cgd,
                        }
                    )

                if args.with_mir_eval:
                    (
                        metrics_mir_eval,
                        t_mir_eval,
                        metrics_sigsep,
                        t_sigsep,
                    ) = instrumented_mir_eval_bss_eval_sources(ref, est)

                    data[-1]["metrics_mireval"] = metrics_mir_eval
                    data[-1]["runtime_mireval"] = t_mir_eval
                    data[-1]["metrics_sigsep"] = metrics_sigsep
                    data[-1]["runtime_sigsep"] = t_sigsep

            print(f"Done {channels}ch {idx}/{len(dataset)}")

    write_config(data, args.output / output_filename)
