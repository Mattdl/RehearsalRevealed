#!/bin/bash

# Add modules in project root path
this_script_path=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_path}/../framework
export PYTHONPATH=$PYTHONPATH:$root_path
echo "Project root added to Python path: '${root_path}'"

MY_PYTHON=python3

$MY_PYTHON grad_norms.py --data mnist --iter 100
$MY_PYTHON grad_norms.py --data cifar --iter 100
$MY_PYTHON grad_norms.py --data min --iter 100
$MY_PYTHON grad_norms_plot.py