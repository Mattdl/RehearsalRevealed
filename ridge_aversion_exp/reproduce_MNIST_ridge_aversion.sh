#!/bin/bash

# Add modules in project root path
this_script_path=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_path}/../framework
export PYTHONPATH=$PYTHONPATH:$root_path
echo "Project root added to Python path: '${root_path}'"

MY_PYTHON=python
pyscript=main.py
exp_name="demo_MNIST"
ds_args="--scenario smnist"

# Paths
mem_size=100

# ER baseline
args="--replay_mode ER --aversion_steps 0 --strategy replay --store_crit rnd --n_seeds 5 --cuda yes --epochs 10 --mem_size $mem_size --bs 10 $exp_name"
$MY_PYTHON "$pyscript" $ds_args $args # Run python file

# ER-step
steps=20 # Do gridsearch per mem_size
args="--replay_mode ERaverse --aversion_steps $steps --strategy replay --store_crit rnd --n_seeds 5 --cuda yes --epochs 10 --mem_size $mem_size --bs 10 $exp_name"
$MY_PYTHON "$pyscript" $ds_args $args # Run python file
