
# ECP-Candle Benchmark - Pilot1/Combo

This benchmark is based on materials from the [ECP-Candle/Benchmarks](https://github.com/ECP-CANDLE/Benchmarks) repository. More precisely it is a simplified replicate of the [Pilot1/Combo](https://github.com/ECP-CANDLE/Benchmarks/tree/master/Pilot1/Combo) benchmark.


## Installation

To use the benchmark follow this example set of instructions:

```python
from deephyper_benchmark import *

install("ECP-Candle/Pilot1/Combo")

load("ECP-Candle/Pilot1/Combo")

from deephyper_benchmark.lib.ecp_candle.pilot1.combo import hpo
```

## Configuration

Different parameters can be set to configure this benchmark.

- Environment variable `DEEPHYPER_BENCHMARK_MAX_EPOCHS` sets the maximum number of training epochs. Defaults to `50`.
- Environment variable `DEEPHYPER_BENCHMARK_TIMEOUT` sets the maximum duration of training in secondes. Defaults to `1800` (i.e., 30 minutes).
- Environment variable `DEEPHYPER_BENCHMARK_MOO` with value `0` or `1` to select if the task should be run with single or multiple objectives. **Defaults to `0` for single-objective**. The objectives are `valid_r2`, `-num_parameters_train`, `-duration_batch_inference`.

## Metadata

The current set of returned metadata is:

- [x] `num_parameters`
- [x] `num_parameters_train`
- [x] `duration_train`: time in secondes taken to train the model (`model.fit(...)`).
- [x] `duration_batch_inference`: time in secondes taken to predict one batch.
- [x] `budget`
- [x] `stopped`
- [x] `train_mse`
- [x] `train_mae`
- [x] `train_r2`
- [x] `train_corr`
- [x] `valid_mse`
- [x] `valid_mae`
- [x] `valid_r2`
- [x] `valid_corr`
- [x] `test_mse`
- [x] `test_mae`
- [x] `test_r2`
- [x] `test_corr`
- [ ] `flops`
- [ ] `latency`
- [x] `lc_train_mse`
- [x] `lc_valid_mse`