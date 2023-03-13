
# PINN Benchmark - Burgers Equation

This benchmark is based on the implementation of Physics-informed Neural networks (https://arxiv.org/abs/1711.10561) on 1D Burgers Equation. 

To use the benchmark follow this example set of instructions:

```python
from deephyper_benchmark import *

install("PINNBench/Burgers")

load("PINNBench/Burgers")

from deephyper_benchmark.lib.PINNBench.Burgers.hpo import run
```