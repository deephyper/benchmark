
# ECP-Candle Benchmark - Pilot1/Combo

This benchmark is based on materials from the [ECP-Candle/Benchmarks](https://github.com/ECP-CANDLE/Benchmarks) repository. More precisely it is a simplified replicate of the [Pilot1/Combo](https://github.com/ECP-CANDLE/Benchmarks/tree/master/Pilot1/Combo) benchmark.

To use the benchmark follow this example set of instructions:

```python
from deephyper_benchmark import *

install("ECP-Candle/Pilot1/Combo")

load("ECP-Candle/Pilot1/Combo")

# required before importing candle to detect the neural network
# framework used
from tensorflow.keras import backend as K

import candle
import combo
```


## Metadata

The current set of returned metadata is:

- [x] `num_parameters`
- [x] `num_parameters_train`
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