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

import os
import numpy as np
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig_output_dir = Path("./figures")
runtime_algo_order = ["mir_eval", "sigsep", "prop_solve", f"prop_cgd_10"]
error_algo_order = ["prop_solve", f"prop_cgd_10"]
cm2in = 0.39
in2cm = 1.0 / cm2in
figsize_cm = [10.0, 5.5]
legend_width_cm = 2.0

col_names = {"error": "$\log_{10}(|\operatorname{error}|)$", "runtime": "Runtime (s)"}
algo_table = {
    "mir_eval": "mir_eval",
    "sigsep": "sigsep",
    "prop_npy_solve_64": "numpy/solve/fl64",
    "prop_npy_solve_32": "numpy/solve/fl32",
    "prop_npy_cgd_10_64": "numpy/CGD10/fl64",
    "prop_npy_cgd_10_32": "numpy/CGD10/fl32",
    "prop_solve_64": "torch/solve/fl64",
    "prop_solve_32": "torch/solve/fl32",
    "prop_cgd_10_64": "torch/CGD10/fl64",
    "prop_cgd_10_32": "torch/CGD10/fl32",
}
algo_order_runtime = list(algo_table.values())
algo_order_error = algo_order_runtime[2:]


def fig_fix_legend(fig, legend_width_cm, legend_kwargs=None):

    cur_figsize = fig.get_size_inches()
    cur_fig_w = cur_figsize[0]
    lgd_w = legend_width_cm * cm2in
    plots_w = cur_fig_w - lgd_w

    fig.tight_layout()

    if legend_kwargs is not None:
        fig.legend(**legend_kwargs)

    fig.subplots_adjust(right=plots_w / cur_fig_w)


def mse(x1, x2):
    return [
        np.mean(np.abs(np.array(x1[k]) - np.array(x2[k])))
        for k in ["sdr", "sir", "sar"]
    ]


def fill_error_table(idx, channels, device, algo, ref, est):
    table = []
    for metric in ["sdr", "sir", "sar"]:
        for x1, x2 in zip(ref[metric], est[metric]):
            table.append([idx, channels, device, algo, metric.upper(), x1 - x2])
    return table


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates figure for runtime from saved data"
    )
    parser.add_argument("datafile", type=Path, help="Path to data file")
    parser.add_argument(
        "datafile_gpu", type=Path, help="Path to data file for GPU results"
    )
    parser.add_argument(
        "datafile_cpu_mt", type=Path, help="Path to data file for GPU results"
    )
    args = parser.parse_args()

    os.makedirs(fig_output_dir, exist_ok=True)

    with open(args.datafile, "r") as f:
        data = json.load(f)

    with open(args.datafile_gpu, "r") as f:
        data_gpu = json.load(f)

    with open(args.datafile_cpu_mt, "r") as f:
        data_mt = json.load(f)

    columns = [
        "id",
        "channels",
        "device",
        "algo",
        "runtime",
    ]
    table = []

    columns_errors = ["id", "channels", "device", "algo", "metric", "error"]
    table_errors = []
    for idx, entry in enumerate(data):

        entry_gpu = data_gpu[idx]
        entry_mt = data_mt[idx]

        ref = entry["metrics_mireval"]
        chan = entry["channels"]
        idx = entry["data_id"]

        # single-threaded
        runtime_mireval = entry["runtime_mireval"]
        runtime_sigsep = entry["runtime_sigsep"]

        runtime_prop_solve_32 = entry["runtime_prop_solve_32"]
        runtime_prop_solve_64 = entry["runtime_prop_solve_64"]
        runtime_prop_cgd_32 = entry["runtime_prop_cgd_32"]
        runtime_prop_cgd_64 = entry["runtime_prop_cgd_64"]

        runtime_prop_npy_solve_32 = entry["runtime_prop_npy_solve_32"]
        runtime_prop_npy_solve_64 = entry["runtime_prop_npy_solve_64"]
        runtime_prop_npy_cgd_32 = entry["runtime_prop_npy_cgd_32"]
        runtime_prop_npy_cgd_64 = entry["runtime_prop_npy_cgd_64"]

        # multi-threaded
        runtime_mireval_mt = entry_mt["runtime_mireval"]
        runtime_sigsep_mt = entry_mt["runtime_sigsep"]

        runtime_prop_solve_32_mt = entry_mt["runtime_prop_solve_32"]
        runtime_prop_solve_64_mt = entry_mt["runtime_prop_solve_64"]
        runtime_prop_cgd_32_mt = entry_mt["runtime_prop_cgd_32"]
        runtime_prop_cgd_64_mt = entry_mt["runtime_prop_cgd_64"]

        runtime_prop_npy_solve_32_mt = entry_mt["runtime_prop_npy_solve_32"]
        runtime_prop_npy_solve_64_mt = entry_mt["runtime_prop_npy_solve_64"]
        runtime_prop_npy_cgd_32_mt = entry_mt["runtime_prop_npy_cgd_32"]
        runtime_prop_npy_cgd_64_mt = entry_mt["runtime_prop_npy_cgd_64"]

        # GPU
        runtime_prop_solve_32_gpu = entry_gpu["runtime_prop_solve_32"]
        runtime_prop_solve_64_gpu = entry_gpu["runtime_prop_solve_64"]
        runtime_prop_cgd_32_gpu = entry_gpu["runtime_prop_cgd_32"]
        runtime_prop_cgd_64_gpu = entry_gpu["runtime_prop_cgd_64"]

        table += [
            [idx, chan, "1CPU", "mir_eval", runtime_mireval],
            [idx, chan, "1CPU", "sigsep", runtime_sigsep],
            [idx, chan, "1CPU", "prop_solve_32", runtime_prop_solve_32],
            [idx, chan, "1CPU", "prop_solve_64", runtime_prop_solve_64],
            [idx, chan, "1CPU", "prop_npy_solve_32", runtime_prop_npy_solve_32],
            [idx, chan, "1CPU", "prop_npy_solve_64", runtime_prop_npy_solve_64],
            [idx, chan, "1CPU", "prop_cgd_10_32", runtime_prop_cgd_32],
            [idx, chan, "1CPU", "prop_cgd_10_64", runtime_prop_cgd_64],
            [idx, chan, "1CPU", "prop_npy_cgd_10_32", runtime_prop_npy_cgd_32],
            [idx, chan, "1CPU", "prop_npy_cgd_10_64", runtime_prop_npy_cgd_64],
            [idx, chan, "8CPU", "mir_eval", runtime_mireval_mt],
            [idx, chan, "8CPU", "sigsep", runtime_sigsep_mt],
            [idx, chan, "8CPU", "prop_solve_32", runtime_prop_solve_32_mt],
            [idx, chan, "8CPU", "prop_solve_64", runtime_prop_solve_64_mt],
            [idx, chan, "8CPU", "prop_npy_solve_32", runtime_prop_npy_solve_32_mt],
            [idx, chan, "8CPU", "prop_npy_solve_64", runtime_prop_npy_solve_64_mt],
            [idx, chan, "8CPU", "prop_cgd_10_32", runtime_prop_cgd_32_mt],
            [idx, chan, "8CPU", "prop_cgd_10_64", runtime_prop_cgd_64_mt],
            [idx, chan, "8CPU", "prop_npy_cgd_10_32", runtime_prop_npy_cgd_32_mt],
            [idx, chan, "8CPU", "prop_npy_cgd_10_64", runtime_prop_npy_cgd_64_mt],
            [idx, chan, "GPU", "prop_solve_32", runtime_prop_solve_32_gpu],
            [idx, chan, "GPU", "prop_solve_64", runtime_prop_solve_64_gpu],
            [idx, chan, "GPU", "prop_cgd_10_32", runtime_prop_cgd_32_gpu],
            [idx, chan, "GPU", "prop_cgd_10_64", runtime_prop_cgd_64_gpu],
        ]

        # errors
        for (method, method2) in [("solve", "solve"), ("cgd", "cgd_10")]:
            for bits in [32, 64]:
                table_errors += fill_error_table(
                    idx,
                    chan,
                    "CPU",
                    f"prop_npy_{method2}_{bits}",
                    ref,
                    entry[f"metrics_prop_npy_{method}_{bits}"],
                )
                for dev, e in zip(["CPU", "GPU"], [entry, entry_gpu]):
                    table_errors += fill_error_table(
                        idx,
                        chan,
                        dev,
                        f"prop_{method2}_{bits}",
                        ref,
                        e[f"metrics_prop_{method}_{bits}"],
                    )

    # Set the style for the plots
    sns.set_theme(context="paper", style="whitegrid", font_scale=0.6)

    # make the figure for speed
    df = pd.DataFrame(table, columns=columns)
    df = df.replace(algo_table)
    df = df.rename(columns=col_names)

    sns.set_palette("colorblind", 6)

    g = sns.catplot(
        data=df,
        col="channels",
        x="device",
        y=col_names["runtime"],
        hue="algo",
        kind="bar",
        height=figsize_cm[1] * cm2in,
        aspect=(figsize_cm[0] / 3) / figsize_cm[1],
        log=True,
        hue_order=algo_order_runtime,
        legend=False,
        errwidth=1.0,
        linewidth=0,
    )

    g.set_titles(col_template="{col_name} channels")
    g.set_xlabels("")
    # g.set(ylim=(-15, None))
    # g.set(yticks=[-15, -10, -5, 0, 2])

    g.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)

    leg_handles = {}
    for ax in g.axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for lbl, hand in zip(labels, handles):
            leg_handles[lbl] = hand

    fig_fix_legend(
        g.fig,
        legend_width_cm,
        legend_kwargs=dict(
            handles=leg_handles.values(),
            labels=leg_handles.keys(),
            fontsize="xx-small",
            frameon=False,
            loc="center right",
            ncol=1,
        ),
    )

    for ext in ["pdf", "png"]:
        plt.savefig(fig_output_dir / f"channels_vs_runtime.{ext}", dpi=300)

    # make the figure for the error
    df_error = pd.DataFrame(table_errors, columns=columns_errors)
    df_error["error"] = df_error["error"].apply(
        lambda x: np.log10(np.maximum(np.abs(x), 1e-20))
    )
    df_error = df_error.replace(algo_table)
    df_error = df_error.rename(columns=col_names)

    sns.set_palette("colorblind", 4)

    g = sns.catplot(
        data=df_error,
        col="metric",
        x="device",
        y=col_names["error"],
        hue="algo",
        kind="box",
        height=figsize_cm[1] * cm2in,
        aspect=(figsize_cm[0] / 3) / figsize_cm[1],
        hue_order=algo_order_error,
        fliersize=0.25,
        linewidth=0.5,
        legend=False,
    )

    g.set_titles(col_template="{col_name}")
    g.set_xlabels("")
    g.set(ylim=(-15, None))
    g.set(yticks=[-15, -10, -5, 0, 2])

    g.fig.tight_layout(pad=0.1, w_pad=1.0, h_pad=0.1)

    leg_handles = {}
    for ax in g.axes.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for lbl, hand in zip(labels, handles):
            leg_handles[lbl] = hand

    fig_fix_legend(
        g.fig,
        legend_width_cm,
        legend_kwargs=dict(
            handles=leg_handles.values(),
            labels=leg_handles.keys(),
            fontsize="xx-small",
            frameon=False,
            loc="center right",
            ncol=1,
        ),
    )

    for ext in ["pdf", "png"]:
        plt.savefig(fig_output_dir / f"metric_vs_error.{ext}", dpi=300)
