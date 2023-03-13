#!/bin/bash

set -x

mkdir build/ && cd build

# clone HPOBench
git clone https://github.com/automl/HPOBench.git
cd HPOBench/

# download data
mkdir data
cd data

# data for tabular benchmarks
wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
tar xf fcnet_tabular_benchmarks.tar.gz

# install other dependencies
cd ..
pip install .[tabular_benchmarks]
pip install git+https://github.com/google-research/nasbench.git@master

#! need to patch the "tabular_benchmarks/__init__.py" by commenting the last line!
git clone https://github.com/automl/nas_benchmarks.git
cd nas_benchmarks
python setup.py install