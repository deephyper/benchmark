#!/bin/bash

set -x

mkdir build/ && cd build

# Clone HPOBench repository
git clone https://github.com/automl/HPOBench.git
cd HPOBench/

# Freeze commit used for the benchmark
git checkout d8b45b1eca9a61c63fe79cdfbe509f77d3f5c779

# Relax the constraint on the Python version required
sed -i '' 's/>=3.6, <=3.10/>=3.6, <3.11/g' setup.py

# Download data
mkdir data && cd data/

# For tabular benchmarks
wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
tar xf fcnet_tabular_benchmarks.tar.gz

# Install other dependencies
cd ..
pip install "."
pip install git+https://github.com/google-research/nasbench.git@master

git clone https://github.com/automl/nas_benchmarks.git
cd nas_benchmarks

# Freeze commit used for the benchmark
git checkout 1b09906ba3f522f15766b75643423acccd9db3a5

# Comment out the last line of the file
sed -i '' '5 s/./#&/' tabular_benchmarks/__init__.py

python setup.py install

