#!/bin/bash
#
# Run all the experiments for the paper
# SDR --- Medium Rare with Fast Computations

# single threaded experiment
OMP_NUM_THREADS=1 python ./benchmark_bsseval.py --with-mir-eval wsj1_2345_db ./output

# multi-threaded experiment
python ./benchmark_bsseval.py --with-mir-eval --with-multithread wsj1_2345_db ./output

# GPU only
python ./benchmark_bsseval.py wsj1_2345_db ./output

# runtime vs ci_sdr on GPU
python ./benchmark_sdr_gpu.py --output ./output

# make figures 1 and 2 in the paper
python ./make_figure_benchmark_bsseval.py \
    output/runtime_accuracy_vs_mir_eval.json \
    output/runtime_accuracy_with_gpu.json \
    output/runtime_accuracy_vs_mir_eval_multithread.json

# make table 1 in the paper
python ./make_table_benchmark_sdr.py ./output/runtime_vs_ci_sdr.json
