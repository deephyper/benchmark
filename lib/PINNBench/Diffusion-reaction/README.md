# PINN Benchmark - Diffusion-reaction Equation
This benchmark is based on **modified** [`PDEBench`](https://github.com/pdebench/PDEBench) and [`DeepXDE`](https://github.com/lululxvi/deepxde). 

## Installation

To install the **modified** `PDEBench`
```
git clone https://github.com/iamyixuan/PDEBench-DH.git
cd PDEBench-DH
python -m pip install -e . 
```
To install `DeepXDE`
```
python -m pip install deepxde
```
It is necessary to configure `DeepXDE` to use `PyTorch` backend. The instructions can be found [here](https://deepxde.readthedocs.io/en/latest/user/installation.html#working-with-different-backends).

## Data Generation
To generate diffusion-reaction data for this PINN benchmark, use data generation script in the modified `PDEBench`.
```
pdebench
├── data_gen
│   ├── gen_diff_react.py
```
```
cd pdebench/data_gen/
python gen_diff_react.py
```
The resulting dataset will be stored in `pdebench/data/` folder. \*Note: replace the dataset path when running the PINN benchmark.

To use the benchmark
```python
from deephyper_benchmark import *

install("PINNBench/Diffusion-reaction")

load("PINNBench/Diffusion-reaction")
```



