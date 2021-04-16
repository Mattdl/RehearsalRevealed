#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2021 KU Leuven                                                 #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 24-03-2021                                                             #
# Author(s): Matthias De Lange, Eli Verwimp                                    #
# E-mail: matthias.delange@kuleuven.be, eli.verwimp@kuleuven.be                #
################################################################################

"""
Main pipeline code of paper "Rehearsal revealed:The limits and merits of revisiting samples in continual learning"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path

from avalanche.evaluation.metrics import StreamConfusionMatrix
from avalanche.logging import TextLogger, TensorboardLogger

import random
import numpy
import torch
import uuid
import datetime
import argparse
from avalanche.benchmarks import SplitMNIST, SplitCIFAR10
from avalanche.evaluation.metrics import ExperienceForgetting, accuracy_metrics, loss_metrics
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies import Naive

from csv_logger import EvalTextLogger

from model import MyMLP, ResNet18
from method import RehRevPlugin
from miniimagenet import SplitMiniImageNet

parser = argparse.ArgumentParser()

# Meta hyperparams
parser.add_argument('exp_name', default=None, type=str, help='id for the experiment.')
parser.add_argument('--save_path', type=str, default='../data/results/', help='save models at the end of training')
parser.add_argument('--output_name', type=str, default='', help='special output name for the results?')
parser.add_argument('--n_seeds', default=5, type=int, help='Nb of seeds to run.')
parser.add_argument('--seed', default=None, type=int, help='Run a specific seed.')
parser.add_argument('--uid', default=None, type=str, help='id for the seed runs.')
parser.add_argument('--resume', default=None, type=str, help='resume in time/uid parentdir')
parser.add_argument('--cuda', default='yes', type=str, help='Disable cuda?')
parser.add_argument('--disable_pbar', default='yes', type=str, help='Disable cuda?')

# Model
parser.add_argument('--model', type=str, default='simple_mlp', choices=['simple_mlp', 'resnet18'],
                    help='model to train.')
parser.add_argument('--hs', type=int, default=400, help='MLP hidden size.')

# Strategy
parser.add_argument('--strategy', type=str, default='replay', choices=['replay'], help='model to train.')
parser.add_argument('--store_crit', type=str, default='rnd', choices=RehRevPlugin.compare_criteria,
                    help='Random storage policy for rehearsal memory.')
parser.add_argument('--replay_mode', type=str, default='ER', choices=RehRevPlugin.modes,
                    help='replay_mode (ER or ERaverse')
parser.add_argument('--mem_size', default=100, type=int, help='Number in rehearsal memory per task.')

# High-loss ridge aversion
parser.add_argument('--aversion_steps', default=0, type=int, help='How many steps to perform on memory.')
parser.add_argument('--aversion_lr', type=float, default=-1,
                    help='Fixed learning rate for the steps on the memory, if -1 use default optimizer lr.')

# Stable SGD for CIFAR/Mini-Imagenet
parser.add_argument('--stable_sgd', default='no', type=str, help='Stable SGD mode.')
parser.add_argument('--lr_decay', default=1, type=float, help='multiplicative factor for lr decay. (inactive=1)')
parser.add_argument('--drop_prob', default=0, type=float,
                    help='Prob to zero out in dropout fro stable SGD.(inactive=0)')

# Optimization Hyperparams
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--epochs', type=int, default=1,
                    help='Number of training epochs/step.')
parser.add_argument('--init_epochs', type=int, default=-1,
                    help='Number of training epochs/step for first experience (-1=same as standard epochs).')
parser.add_argument('--bs', type=int, default=10,
                    help='Minibatch size.')

# Dataset
parser.add_argument('--scenario', type=str,
                    choices=['smnist', 'CIFAR10', 'miniimgnet'], default='smnist',
                    help='Choose between Permuted MNIST, Split MNIST.')
parser.add_argument('--until_task', default=5, type=int,
                    help='Task nb to cut off training.'
                         'e.g. 5 -> Learns/evals only for Task 0 -> 4 (5 tasks total)')
parser.add_argument('--dset_rootpath', default='./data', type=str,
                    help='Root path of the downloaded dataset.')  # Mini Imagenet


def main():
    args = parser.parse_args()
    args.cuda = args.cuda == 'yes'
    args.disable_pbar = args.disable_pbar == 'yes'
    args.stable_sgd = args.stable_sgd == 'yes'
    print(f"args={vars(args)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f'Using device: {device}')

    # unique identifier
    uid = uuid.uuid4().hex if args.uid is None else args.uid
    now = str(datetime.datetime.now().date()) + "_" + ':'.join(str(datetime.datetime.now().time()).split(':')[:-1])
    runname = 'T={}_id={}'.format(now, uid) if not args.resume else args.resume

    # Paths
    setupname = [args.strategy, args.exp_name, args.model, args.scenario]
    parentdir = os.path.join(args.save_path, '_'.join(setupname))
    results_path = Path(os.path.join(parentdir, runname))
    results_path.mkdir(parents=True, exist_ok=True)
    tb_log_dir = os.path.join(results_path, 'tb_run')  # Group all runs

    # Eval results
    eval_metric = 'Top1_Acc_Stream/eval_phase/test_stream'
    eval_results_dir = results_path / eval_metric.split('/')[0]
    eval_results_dir.mkdir(parents=True, exist_ok=True)

    eval_result_files = []  # To avg over seeds
    seeds = [args.seed] if args.seed is not None else list(range(args.n_seeds))
    for seed in seeds:
        # initialize seeds
        print("STARTING SEED {}/{}".format(seed, len(seeds) - 1))

        set_seed(seed)

        # create scenario
        if args.scenario == 'smnist':
            inputsize = 28 * 28
            scenario = SplitMNIST(n_experiences=5, return_task_id=False, seed=seed,
                                  fixed_class_order=[i for i in range(10)])
        elif args.scenario == 'CIFAR10':
            scenario = SplitCIFAR10(n_experiences=5, return_task_id=False, seed=seed,
                                    fixed_class_order=[i for i in range(10)])
            inputsize = (3, 32, 32)
        elif args.scenario == 'miniimgnet':
            scenario = SplitMiniImageNet(args.dset_rootpath, n_experiences=20, return_task_id=False, seed=seed,
                                         fixed_class_order=[i for i in range(100)])
            inputsize = (3, 84, 84)
        else:
            raise ValueError("Wrong scenario name.")
        print(f"Scenario = {args.scenario}")

        if args.model == 'simple_mlp':
            model = MyMLP(input_size=inputsize, hidden_size=args.hs)
        elif args.model == 'resnet18':
            if not args.stable_sgd:
                assert args.drop_prob == 0
            model = ResNet18(inputsize, scenario.n_classes, drop_prob=args.drop_prob)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        # Paths
        eval_results_file = eval_results_dir / f'seed={seed}.csv'

        # LOGGING
        tb_logger = TensorboardLogger(tb_log_dir=tb_log_dir, tb_log_exp_name=f'seed={seed}.pt')  # log to Tensorboard
        print_logger = TextLogger() if args.disable_pbar else InteractiveLogger()  # print to stdout
        eval_logger = EvalTextLogger(metric_filter=eval_metric, file=open(eval_results_file, 'a'))
        eval_result_files.append(eval_results_file)

        # METRICS
        eval_plugin = EvaluationPlugin(
            accuracy_metrics(experience=True, stream=True),
            loss_metrics(minibatch=True, experience=True),
            ExperienceForgetting(),  # Test only
            StreamConfusionMatrix(num_classes=scenario.n_classes, save_image=True),

            # LOG OTHER STATS
            # timing_metrics(epoch=True, experience=False),
            # cpu_usage_metrics(experience=True),
            # DiskUsageMonitor(),
            # MinibatchMaxRAM(),
            # GpuUsageMonitor(0),
            loggers=[print_logger, tb_logger, eval_logger])

        plugins = None
        if args.strategy == 'replay':
            plugins = [RehRevPlugin(n_total_memories=args.mem_size,
                                    mode=args.replay_mode,  # STEP-BACK
                                    aversion_steps=args.aversion_steps,
                                    aversion_lr=args.aversion_lr,
                                    stable_sgd=args.stable_sgd,  # Stable SGD
                                    lr_decay=args.lr_decay,
                                    init_epochs=args.init_epochs  # First task epochs
                                    )]

        # CREATE THE STRATEGY INSTANCE (NAIVE)
        strategy = Naive(model, optimizer, criterion,
                         train_epochs=args.epochs, device=device,
                         train_mb_size=args.bs, evaluator=eval_plugin,
                         plugins=plugins
                         )

        # train on the selected scenario with the chosen strategy
        print('Starting experiment...')
        for experience in scenario.train_stream:
            if experience.current_experience == args.until_task:
                print("CUTTING OF TRAINING AT TASK ", experience.current_experience)
                break
            else:
                print("Start training on step ", experience.current_experience)

            strategy.train(experience)
            print("End training on step ", experience.current_experience)
            print('Computing accuracy on the test set')
            res = strategy.eval(scenario.test_stream[:args.until_task])  # Gathered by EvalLogger

    final_results_file = eval_results_dir / f'seed_summary.pt'
    stat_summarize(eval_result_files, final_results_file)
    print(f"[FILE:TB-RESULTS]: {tb_log_dir}")
    print(f"[FILE:FINAL-RESULTS]: {final_results_file}")
    print("FINISHED SCRIPT")


def stat_summarize(stat_files, save_file=None):
    # To aggregate Tensorboard results, also see: https://github.com/Spenhouet/tensorboard-aggregator
    import csv
    from collections import OrderedDict

    print("Taking avg of {} results: {}".format(len(stat_files), stat_files))

    # Collect all per row
    res_collect = OrderedDict({})
    for stat_file in stat_files:  # Keep row,col dims -> collect over files
        with open(stat_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row_idx, row in enumerate(reader):
                for col_idx, (k, v) in enumerate(row.items()):  # Vals
                    if k not in res_collect:
                        res_collect[k] = [[]]
                    if row_idx >= len(res_collect[k]):
                        res_collect[k].append([])
                    res_collect[k][row_idx].append(float(v))  # Convert to float

    # Avg/std per row
    # assert all lengths equal
    lencheck = None
    mean_over_files = OrderedDict({})  # List of per step means/avgs
    std_over_files = OrderedDict({})  # List of per step means/avgs
    for col_name, row_listvals in res_collect.items():  # lists over the files
        if lencheck is None:
            lencheck = len(v)
        else:
            assert len(v) == lencheck, "NOT ALL RESULTS HAVE BEEN OBTAINED FOR ALL STEPS, Can't avg!"

        mean_over_files[col_name] = []
        std_over_files[col_name] = []
        for rowlist in row_listvals:  # e.g. avg_acc metric -> All values of step 1
            step_values = torch.tensor(rowlist)  # list to tensor
            mean = step_values.mean() * 100
            std = step_values.std() * 100

            mean_over_files[col_name].append(mean)
            std_over_files[col_name].append(std)

    # Now we have a dict with per metric-key: a list of avg/std values over all the steps: Final = last step
    for k1, v1, in mean_over_files.items():
        v2 = std_over_files[k1]
        if 'step_' in k1: continue  # Don't need to avg on the steps
        print(f"[METRIC: {k1}]=" + '\t'.join(["{:.3f}\pm{:.3f}".format(mean, std) for mean, std in zip(v1, v2)]))
        print("[FINAL AVG ACC]={:.5f}\pm{:.5f}".format(v1[-1], v2[-1]))

    if save_file is not None:
        try:
            torch.save({'mean': mean_over_files, 'std': std_over_files}, save_file)
        except Exception as e:
            print(f"NOT SAVING SUMMARY: {e}")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


if __name__ == '__main__':
    main()
