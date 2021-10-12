import os
import numpy as np
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig_output_dir = Path("./figures")
cm2in = 0.39

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates figure for runtime of fast bss eval vs ci-sdr"
    )
    parser.add_argument("datafile", type=Path, help="Path to data file")
    args = parser.parse_args()

    os.makedirs(fig_output_dir, exist_ok=True)

    with open(args.datafile, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    df = df.rename(
        columns={
            "signal_length": "Signal length",
            "filter_length": "Filter length",
            "runtime": "Runtime (s)",
            "channels": "Channels",
            "algo": "Package",
        }
    )

    df = df.replace(
        {
            "cisdr": "CI-SDR",
            "solve": "Proposed (solve)",
            "cgd": "Proposed (CGD)",
        }
    )

    sns.set_theme(context="paper", style="whitegrid")

    g = sns.catplot(
        data=df,
        row="Signal length",
        col="Filter length",
        x="Channels",
        hue="Package",
        y="Runtime (s)",
        kind="bar",
        margin_titles=True,
        legend=True,
        aspect=1.0,
        height=8.5 * cm2in / 2.0,
        log=False,
    )

    g.set_titles(row_template="{row_name:.0f} s", col_template="{col_name} taps")

    # g.fig.tight_layout()

    # g.fig.legend(fontsize="x-small")

    plt.savefig(fig_output_dir / "runtime_vs_cisdr.pdf")

    # Create the latex table
    table_ms = df.rename(columns={"Runtime (s)": "Runtime (ms)"}).pivot_table(
        values=["Runtime (ms)", "speedup"],
        index=["Channels", "Package"],
        columns=["Signal length", "Filter length"],
    )
    table_ms["Runtime (ms)"] = table_ms["Runtime (ms)"].apply(lambda x: x * 1000)
    table_ms["speedup"] = table_ms["speedup"].applymap(lambda x: f"(${x:.0f}\times$)")

    table_ms = table_ms.reorder_levels([2, 1, 0], axis="columns").sort_index(
        axis="columns"
    )

    print(table_ms.to_latex(float_format="%.0f", escape=False))
    print()
    print(table_ms)
