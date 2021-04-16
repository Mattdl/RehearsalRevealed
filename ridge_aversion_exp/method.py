################################################################################
# Copyright (c) 2021 KU Leuven                                                 #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-03-2021                                                             #
# Author(s): Matthias De Lange, Eli Verwimp                                    #
# E-mail: matthias.delange@kuleuven.be, eli.verwimp@kuleuven.be                #
################################################################################
from avalanche.training.plugins import StrategyPlugin
from torch.utils.data import random_split, ConcatDataset, DataLoader
import torch
import random
import copy
from torch.utils.data import Subset


class RehRevPlugin(StrategyPlugin):
    """
    Rehearsal Revealed: replay plugin.
    Implements two modes: Classic Experience Replay (ER) and Experience Replay with Ridge Aversion (ERaverse).
    """

    modes = ['ER', 'ERaverse']
    compare_criteria = ['rnd']

    def __init__(self, n_total_memories=2000, num_tasks=5, mode='ERaverse', init_epochs=-1,
                 stable_sgd=False, lr_decay=1, aversion_steps=0, aversion_lr=-1):
        """
        
        :param n_total_memories: The maximal number of input samples to store in total. 
        :param num_tasks:        The number of tasks being seen in the scenario.
        :param mode:             'ER'=regular replay, 'ERaverse'=Replay with Ridge Aversion.
        :param aversion_steps:   How many update steps to perform only on the buffer after regular ER converging.
        :param stable_sgd:       Enable Stable-SGD (including momentum, and learning rate decay).
        :param lr_decay:         When Stable-SGD enabled, the learning rate decay factor.
        :param init_epochs:      Number of epochs for the first experience/task.
        :param aversion_lr:      When defined, this is a fixed learning rate for the aversion_steps.
        """
        super().__init__()
        self.mode = mode
        assert self.mode in self.modes

        # First task epochs
        self.init_epochs = init_epochs

        # Stable regime
        self.lr_decay = lr_decay
        self.stable_sgd = stable_sgd
        if not self.stable_sgd:
            assert self.lr_decay == 1

        # Memory
        self.n_total_memories = n_total_memories
        self.n_task_memories = self.n_total_memories / num_tasks  # Initially

        self.ext_mem = {}  # a Dict<task_id, Dataset>
        self.rm_add = None
        self.dyn_mem = True  # True: Divide all available buffer space (False = preallocate for all tasks)

        # MODE OVERWRITES
        self.aversion_steps = 0
        if self.mode == 'ERaverse':
            self.aversion_lr = aversion_lr
            self.compare_criterion = 'rnd'  # rnd retrieval
            self.aversion_steps = aversion_steps
            assert self.aversion_steps > 0, "Can't do lack-behind on buffer with 0 steps."

        print(f"[METHOD CONFIG] n_total_mems={self.n_total_memories} ")

    def before_training_exp(self, strategy, **kwargs):
        """Set number of epochs for first experience (warmup) and set optimizer for Stable-SGD."""
        curr_task_id = strategy.experience.current_experience  # Starts at 0

        # Set first exp epochs
        if self.init_epochs > 0:  # First experience epochs
            if curr_task_id == 0:
                strategy.orig_epochs = strategy.train_epochs
                strategy.train_epochs = self.init_epochs
                print(f"FIRST TASK, TRAINING {strategy.train_epochs} EPOCHS")
            else:
                strategy.train_epochs = strategy.orig_epochs  # reset
                print(f"REMAINING TASKS, TRAINING {strategy.train_epochs} EPOCHS")

        if self.stable_sgd:
            # From original Stable-SGD code
            lr = max(strategy.optimizer.defaults['lr'] * self.lr_decay ** (curr_task_id + 1), 0.00005)  # STABLE SGD
            strategy.optimizer = torch.optim.SGD(strategy.model.parameters(), lr=lr, momentum=0.8)  # Also momentum
            print(f"STABLE SGD: task {curr_task_id} lr={lr}")

    def before_forward(self, strategy, **kwargs):
        """Add samples from rehearsal memory to current batch."""
        x_s, y_s = self.retrieve_exemplar_batch(strategy, nb=strategy.train_mb_size)
        if x_s is not None:  # Add
            assert y_s is not None
            strategy.mb_x = torch.cat([strategy.mb_x, x_s])
            strategy.mb_y = torch.cat([strategy.mb_y, y_s])

    def retrieve_exemplar_batch(self, strategy, nb=None):
        """
        Wrapper to retrieve a batch of exemplars from the rehearsal memory
        :param nb: Number of memories to return
        :return: input-space tensor, label tensor
        """
        dev = strategy.mb_x.device
        n_exemplars = strategy.train_mb_size if nb is None else nb  # Equal amount as batch: Last batch can contain fewer!!
        ret_x = None
        ret_y = None
        if self.n_total_memories > 0 and len(self.ext_mem) > 0:  # Only sample if there are stored
            new_dset = self._retrieve_exemplar_batch(n_exemplars)
            dloader = DataLoader(new_dset, batch_size=len(new_dset), pin_memory=True, shuffle=False)

            for sample in dloader:
                x_s, y_s = sample[0].to(dev), sample[1].to(dev)
                ret_x = x_s if ret_x is None else torch.cat([ret_x, x_s])
                ret_y = y_s if ret_y is None else torch.cat([ret_y, y_s])
        return ret_x, ret_y

    def _retrieve_exemplar_batch(self, n_samples):
        """
        Retrieve a batch of exemplars from the rehearsal memory.
        First sample indices for the available tasks at random, then actually extract from rehearsal memory.
        There is no resampling of exemplars.

        :param n_samples: Number of memories to return
        :return: input-space tensor, label tensor

        """
        assert n_samples > 0, "Need positive nb of samples to retrieve!"

        # Determine how many mem-samples available
        q_total_cnt = 0  # Total samples
        free_q = {}  # idxs of which ones are free in mem queue
        tasks = []
        for t, mem in self.ext_mem.items():
            mem_cnt = len(mem)  # Mem cnt
            free_q[t] = list(range(0, mem_cnt))  # Free samples
            q_total_cnt += len(free_q[t])  # Total free samples
            tasks.append(t)

        # Randomly sample how many samples to idx per class
        free_tasks = copy.deepcopy(tasks)
        tot_sample_cnt = 0
        sample_cnt = {c: 0 for c in tasks}  # How many sampled already
        max_samples = n_samples if q_total_cnt > n_samples else q_total_cnt  # How many to sample (equally divided)
        while tot_sample_cnt < max_samples:
            t_idx = random.randrange(len(free_tasks))
            t = free_tasks[t_idx]  # Sample a task

            if sample_cnt[t] >= len(self.ext_mem[t]):  # No more memories to sample
                free_tasks.remove(t)
                continue
            sample_cnt[t] += 1
            tot_sample_cnt += 1

        # Actually sample
        s_cnt = 0
        subsets = []
        for t, t_cnt in sample_cnt.items():
            if t_cnt > 0:
                # Set of idxs
                cnt_idxs = torch.randperm(len(self.ext_mem[t]))[:t_cnt]
                sample_idxs = cnt_idxs.unsqueeze(1).expand(-1, 1)
                sample_idxs = sample_idxs.view(-1)

                # Actual subset
                s = Subset(self.ext_mem[t], sample_idxs.tolist())
                subsets.append(s)
                s_cnt += t_cnt
        assert s_cnt == tot_sample_cnt == max_samples
        new_dset = ConcatDataset(subsets)

        return new_dset

    def after_training_exp(self, strategy, **kwargs):
        """ After training we update the external memory with the patterns of the current training batch/task.
        For ERaverse mode, we also train solely on the rehearsal memory."""

        # Take a few steps back
        if self.mode == 'ERaverse':
            if self.n_total_memories > 0 and len(self.ext_mem) > 0:  # Only sample if there are stored
                self.ridge_averse_updates(strategy)

        # Last task needs no mem update anymore
        if strategy.experience.current_experience <= strategy.experience.scenario.n_experiences - 1:
            self.store_exemplars(strategy)

    def ridge_averse_updates(self, strategy):
        """ Update only on the rehearsal memory to step away from the bad loss-ridge approximation."""
        # Set fixed step optimizer
        if self.aversion_lr > 0:
            optimizer = torch.optim.SGD(strategy.model.parameters(), lr=self.aversion_lr)  # Optimizer with fixed lr
        else:
            optimizer = strategy.optimizer  # Just take strategy optimizer

        print("Ridge aversion updates")
        optimizer.zero_grad()
        for step in range(self.aversion_steps):
            # Fills x/y with buffer
            strategy.mb_x, strategy.mb_y = self.retrieve_exemplar_batch(strategy, nb=strategy.train_mb_size * 2)

            strategy.logits = strategy.model(strategy.mb_x)  # Forward
            lack_loss = strategy.criterion(strategy.logits, strategy.mb_y)  # Loss & Backward

            lack_loss.backward()  # Backprop
            optimizer.step()  # Optimization step

    def store_exemplars(self, strategy):
        """ Store exemplars from the current experience in the rehearsal memory."""
        curr_task_id = strategy.experience.current_experience

        # Update memory sizes of other tasks + new task-mem size
        if self.dyn_mem:
            self.update_dyn_memsizes(curr_task_id)

        # Add current task data to buffer
        curr_data = strategy.experience.dataset  # The new task data, we select it up front
        subset_size = min(self.n_task_memories, len(curr_data))

        print("[STORING SAMPLES] Sample selection started")
        if self.compare_criterion == 'rnd':  # Just take a random selection of samples
            # We recover it using the random_split method and getting rid of the second split.
            rnd_set, _ = random_split(
                curr_data, [subset_size, len(curr_data) - subset_size]
            )
            self.rm_add = ConcatDataset([rnd_set])
        else:
            raise NotImplementedError()

        # replace patterns in random memory
        if curr_task_id not in self.ext_mem:
            self.ext_mem[curr_task_id] = self.rm_add
        else:  # Add rm_add and keep a random subset of those that were present
            rem_len = len(self.ext_mem[curr_task_id]) - len(self.rm_add)
            _, saved_part = random_split(self.ext_mem[curr_task_id],
                                         [len(self.rm_add), rem_len]
                                         )
            self.ext_mem[curr_task_id] = ConcatDataset([saved_part, self.rm_add])

    def update_dyn_memsizes(self, new_task_id):
        """ Updates the rehearsal memory size per task dynamically.
        For N tasks, each task has 1/Nth of the full memory capacity.
        Tasks with too many samples drop samples by random selection.
        """
        assert self.dyn_mem
        n_seen_tasks = len(self.ext_mem)
        if new_task_id not in self.ext_mem:
            n_seen_tasks += 1
        n_mem = int(self.n_total_memories / n_seen_tasks)
        if n_mem == self.n_task_memories:  # Nothing changed
            return
        print('-' * 80)
        print("Updating memory capacity from {} -> {} ({} seen tasks)".format(
            self.n_task_memories, n_mem, n_seen_tasks))
        self.n_task_memories = n_mem
        updated_mem = {}
        for c, cdset in self.ext_mem.items():  # Cutoff dataset
            orig_size = len(cdset)
            if orig_size > self.n_task_memories:
                cutoff_data, _ = random_split(
                    cdset, [self.n_task_memories, len(cdset) - self.n_task_memories]
                )
            else:
                cutoff_data = cdset
            updated_mem[c] = cutoff_data
            print("Memory of task {} cutoff from {} -> {}".format(c, orig_size, len(updated_mem[c])))

        self.ext_mem = updated_mem
