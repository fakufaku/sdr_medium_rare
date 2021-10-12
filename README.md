SDR &emdash; Medium Rare with Fast Computations
===============================================

This repository contains the code to reproduce some of the experiments of
in the paper [SDR &emdash; Medium Rare with Fast Computation](arxiv_link).
This is essentially a benchmark of the
[fast\_bss\_eval](https://github.com/fakufaku/fast_bss_eval) Python package for
the evaluation of blind source separation algorithms.

**Abstract** &emdash; We revisit the widely used bss eval metrics for source
separation with an eye out for performance. We propose a fast algorithm fixing
shortcomings of publicly available implementations. First, we show that the
metrics are fully specified by the squared cosine of just two angles between
estimate and reference subspaces. Second, large linear systems are involved.
However, they are structured, and we apply a fast iterative method based on
conjugate gradient descent. The complexity of this step is thus reduced by a
factor quadratic in the distortion filter size used in bss eval, usually 512.
In experiments, we assess speed and numerical accuracy. Not only is the loss of
accuracy due to the approximate solver acceptable for most applications, but
the speed-up is up to two orders of magnitude in some, not so extreme, cases.
We confirm that our implementation can train neural networks, and find that
longer distortion filters may be beneficial.

Author
------

[Robin Scheibler](mailto:robin[dot]scheibler[at]linecorp.com)

Quick Start
-----------

Assuming use of [anaconda](https://www.anaconda.com/products/individual)

```
git clone <repo>
cd <repo>
conda env create -f environment.yml
./run_experiments.sh
```

This will produce some data in the `output` folder and figures in the `figures` folder.

fast\_bss\_eval Benchmark Result
--------------------------------

We compare to [mir\_eval](https://github.com/craffel/mir_eval) and [sigsep](https://github.com/sigsep/bsseval) implementations.

### Speed

<img src="./figures/channels_vs_runtime.pdf">

### Numerical Accuracy vs mir\_eval

<img src="./figures/metric_vs_error.pdf">

License
-------

2021 (c) Robin Scheibler, LINE Corporation

This code is released under [MIT License](https://opensource.org/licenses/MIT).
