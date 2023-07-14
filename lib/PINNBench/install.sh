#!/bin/bash


mkdir build/ && cd build/
# install modified PDEBench
git clone https://github.com/iamyixuan/PDEBench-DH.git
cd PDEBench-DH/
python -m pip install -e . 

# generate data
mkdir ./pdebench/data/
cd ./pdebench/data_gen/
python gen_diff_react.py
