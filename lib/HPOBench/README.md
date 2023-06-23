# HPOBench

This set of benchmarks provides precomputed evaluations of neural networks for hyperparameter optimization. This is particularly useful to compare optimization schemes without training neural networks for real.

## Installation

To install this benchmark the following command can be run:

```console
python -c "import deephyper_benchmark as dhb; dhb.install('HPOBench/tabular/navalpropulsion');"
```

For detailed information see the [NeurIPS paper describing the HPOBench benchmark](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/93db85ed909c13838ff95ccfa94cebd9-Abstract-round2.html).

## Configuration

Different parameters can be set to configure this benchmark.

- Environment variable `DEEPHYPER_BENCHMARK_SIMULATE_RUN_TIME` with boolean value in `[0, 1]` which set if the "recorded training time" of each training epoch should be simulated or not. **Defaults to `0`.**
- Environment variable `DEEPHYPER_BENCHMARK_PROP_REAL_RUN_TIME` with real value $> 0$. It represents the proportion of the "recorded training time" which should be used for simulation. For example, `1.0` (default) corresponds to 100% of the training time while `0.1` corresponds to 10% of the training time. **Defaults to `1.0`.**
- Environment variable `DEEPHYPER_BENCHMARK_TASK` to select the task on which to run the benchmark with values in:
    - **`navalpropulsion` (Default)**
    - `parkinsonstelemonitoring`
    - `proteinstructure`
    - `slicelocalization`
- Environment variable `DEEPHYPER_BENCHMARK_MOO` with value `0` or `1` to select if the task should be run with single or multiple objectives. **Defaults to `0` for single-objective**.