### `grad_norms.py`
Script for the gradient norms experiments. Script will train a model sequentially for 5 tasks, and calculate the 
gradients on the memory and test set after 1, 2 and 5 tasks. This is stored at `./results/[DATA]/grad_norms
_[RANDOM_IDX].npy` as a numpy array with dimensions (iters x 6). The first three elements are on the
testset, the next three on the memory. Upon running the plot script, all files  with random indices will be merged to 
the file `grad_norms_merged.npy`. This file contains all stored runs. 

_Note: uncomment the necessary lines in the plot line to obtain the 
desired plots._

Options:
* `--iter ITER `: How many iterations should be done (same as running script multiple times)
* `--data DATA`: Which dataset to use (mnist, cifar, min)