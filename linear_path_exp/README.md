### `linear_path.py`
Script for the linear path plots. Script will first train a model on task 1. Then from this model two new models are 
trained with different buffers **and** data order up to task 5. 16 paths are calculated:
* From w1 to w2, T1 test data
* From w1 to w2, T1 train data
* From w1 to w2, on the first memory
* From w1 to w2, on the second memory
* From w2 to w2' T1 test
* From w2 to W2' T1 + T2 test
* From w2 to w2', first memory
* From w2 to w2', second memory
* From w1 to w5, T1 test data
* From w1 to w5, T1 train data
* From w1 to w5, on the first memory
* From w1 to w5, on the second memory
* From w5 to w5' T1 test
* From w5 to W5' T{1...5} test
* From w5 to w5', first memory
* From w5 to w5', second memory

Each path is by default sampled ten times. This can be changed in the script. The file `linear_path_idx_[F_IDX].csv` 
stores an identifier for each path in the right category. The raw loss values are in `linear_path_raw_[F_IDX].csv`,
where each path is identifiable by its ID in the first row.
  
Options:
* `--data DATASET`: which data to train on (mnist, cifar, min)
* `--f_idx F_IDX`: Index to append to file. Files should be merged manually to file with F_IDX=m before plotting.
  
