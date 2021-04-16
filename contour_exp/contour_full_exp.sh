#!/bin/bash

# Add modules in project root path
this_script_path=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_path}/../framework
export PYTHONPATH=$PYTHONPATH:$root_path
echo "Project root added to Python path: '${root_path}'"

data=mnist
MY_PYTHON=python3

$MY_PYTHON ../framework/util/train_model.py --data $data --tasks 1 --buffer --buf_size 100 --save w1t
$MY_PYTHON ../framework/util/train_model.py --data $data --tasks 2 --buffer --init w1t --save w2t --save_path --test 50
$MY_PYTHON ../framework/util/train_model.py --data $data --tasks 2 --init w1t --save w2t_FT

grid_size=50
samples=100

$MY_PYTHON mode_connectivity.py w1t w2t_FT w2t --data $data --tasks 12 --buffer w1t --path w2t --save test --grid $grid_size --size $samples

$MY_PYTHON mode_connectivity_plot.py test_${grid_size}_${samples} --data $data --path w2t

