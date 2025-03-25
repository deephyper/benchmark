# LCBench

✅ **Benchmark ready to be used.**

This benchmark is only compatible with random search and only interesting to benchmark multi-fidelity algorithms that do not consider hyperparameter optimization. Candidate learners are represented by an id.

## Installation

To install the `LCBench` benchmark, run:
```console
python -c "import deephyper_benchmark as dhb; dhb.install('LCBench');"
```

Run the hyperparameter search
```
import deephyper_benchmark as dhb
dhb.load("LCBench")
from deephyper_benchmark.lib.lcbench import hpo
```