### Contour Experiments

The scripts in this folder are used to make the contour plots of our paper. See 
below for the explanation of the main script. The `_plot.py` script plots the result.
For a full experiment test, see the included bash script.

### `mode_connectivity.py`
This script takes three stored models and uses these to create a 2D  hyperplane in parameter space. Then it will iterate
over this hyperplane and calculate the loss for the given datasets at these points. The results will be stored as numpy 
arrays, which can then later be used to create plots. The result array has the shape (GRID X GRID X 2 X #TESTSETS). The 
third dimention differs between accuracies and loss values. The testsets are an array of the tasks and buffers provided, 
in that order.
Options are:
* `[NAME_1, NAME_2, NAME_3]`: required option, models to use to form hyperplane. Should be stored at `./results/[DATA]/`
  as torch state dicts.
* `--data DATASET`: Data to use: mnist, cifar or min *(=MiniImageNet)*
* `--tasks TASKS`: which tasks to use for loss testing. Ex: 1 for task 1, 12 for task 1 and 2
* `--size SIZE`: Number of sample to use for a dataset to calculate loss. Use all samples by setting to 0, which is the
  default. Mainly used for CIFAR10, to limit the calculation time. Should be large enough to get accurate results.
  If less than SIZE samples are available, the limit will be ignored. Experiments used SIZE=2500.
* `--buffer [BUFFER ...]`: Names of the buffers to test on. Will be after the tasks in the test order. Should be stored 
  at `./results/[DATA]/` as pickled objects.
* `--path [PATH ...]`: If intermediate models are available, use this option with the name of the final model (i.e. the 
  name of the folder of the intermediate models) to calculate and store the projections to the hyperplane. This will be 
  stored in `./models/[DATA]/` as a two-dimensional numpy array.
* `--grid GRID`: set the size of the grid. Number of points to test equals GRID * GRID
* `--start`: The relative coordinate to start the grid, in both x and y direction. Default is -0.5, which means half the
  distance between model 1 and model 2 in the x direction, and half the y-distance between model 3 and model 1 in the y
  direction.
* `--width WIDTH`: Relative width of the grid to test. Default is 2.
* `--save SAVE`: Name to store the results. Result will be at `./results/[DATA]/[SAVE]_[GRID]_[SIZE].npy`
* `--threads THREADS`: Number of threads to use in parallell. Sometimes gives CUDA memory errors (Cuda can be turned of 
  inside the script.)