
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