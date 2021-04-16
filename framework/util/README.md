### `train_model.py`
This script is used to train and store models, with different options. Relevant options are:
* `--data DATASET`: Data to use. mnist, cifar or min *(=MiniImageNet)* 
* `--tasks TASKS`: which tasks to use in training. Ex: 1 for task 1, 12 for task 1 and 2
* `--joint`: Train tasks jointly rather than sequentially
* `--epochs EPOCHS`: number of epochs to train per task
* `--buffer`: Use memory (i.e. use experience replay)
* `--buf_size BUF_SIZE`: set size of the buffer to use, ignored if `--buffer` is not set
* `--save NAME`: saves the final model as a torch state dict at `./results/[DATA]/[NAME].pt`. If `--buffer` is set,
  the memory will be stored in the same directory as a pickled object as`[NAME]_buffer.pkl`. This will also store a
  `.json` file with the same name listing all options that were set to train the model.
* `--init NAME`: This will load an initial model rather than starting training from scratch. If `--buffer`is set the 
  stored memory with the same name will also be loaded. Unless `--buf_name` is provided, then this memory will be used.
* `--save_path`: This will store intermediate models at `./results/[DATA]/[NAME]/model_[ITERATION].pt`. Nb of iterations 
  between saving can be set with `--test`
* `--test TEST`: How many steps to do before saving if `--save_path` is set and after how many steps the model is tested.
* `--help`: Show other options