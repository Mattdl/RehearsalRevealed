#!/bin/bash

# Add modules in project root path
this_script_path=$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P) # Change location to current script
root_path=${this_script_path}/../framework
export PYTHONPATH=$PYTHONPATH:$root_path
echo "Project root added to Python path: '${root_path}'"

MY_PYTHON=python3
M=100
DATA=mnist

for i in {1..M}
do
  $MY_PYTHON linear_path.py --data $DATA
done

$MY_PYTHON linear_path_plot.py --data $DATA

