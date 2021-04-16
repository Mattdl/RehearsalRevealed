################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
################################################################################

from torchvision import transforms
from torch.utils.data.dataset import Dataset

from avalanche.benchmarks.generators import nc_scenario
from avalanche.benchmarks.datasets.mini_imagenet.mini_imagenet import MiniImageNetDataset
from avalanche.benchmarks.utils import train_test_transformation_datasets

_default_train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

_default_test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])


def SplitMiniImageNet(root_path, n_experiences=20, return_task_id=False, seed=0,
                      fixed_class_order=None,
                      train_transform=_default_train_transform,
                      test_transform=_default_test_transform,
                      preprocessed=True):
    """
    Creates a CL scenario using the Mini ImageNet dataset.
    If the dataset is not present in the computer the method automatically
    download it and store the data in the data folder.

    For our experiments we downloaded Imagenet and outputted the preprocessed MiniImagenet file in
    {root_path}/miniImageNet.pkl.
    The pickled file is then read out using the {root_path}/miniimagenet_split_20.txt.

    :param preprocessed: True if using the preprocessed pickle file with all MiniImagenet data preprocessed.
    :param root_path: Root path of the downloaded dataset.
    :param n_experiences: The number of experiences in the current scenario.
    :param return_task_id: if True, for every experience the task id is returned
        and the Scenario is Multi Task. This means that the scenario returned
        will be of type ``NCMultiTaskScenario``. If false the task index is
        not returned (default to 0 for every batch) and the returned scenario
        is of type ``NCSingleTaskScenario``.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param test_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.

    :returns: A :class:`NCMultiTaskScenario` instance initialized for the the
        MT scenario using CIFAR10 if the parameter ``return_task_id`` is True,
        a :class:`NCSingleTaskScenario` initialized for the SIT scenario using
        CIFAR10 otherwise.
        """

    if preprocessed:
        (train_set, test_set), task_labels = _get_mini_imagenet_dataset_preprocessed(
            root_path, train_transform, test_transform)
    else:
        train_set, test_set, task_labels = _get_mini_imagenet_dataset(
            root_path, train_transform, test_transform)
        task_labels = fixed_class_order

    if return_task_id:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=task_labels,
            class_ids_from_zero_in_each_exp=True)
    else:
        return nc_scenario(
            train_dataset=train_set,
            test_dataset=test_set,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=task_labels)


def _get_mini_imagenet_dataset_preprocessed(path, train_transformation, test_transformation):
    """
    Use preprocessed train/val/test split.
    Download: https://github.com/yaoyao-liu/mini-imagenet-tools
    """
    train_set, test_set, task_labels = get_split_mini_imagenet(path)

    return train_test_transformation_datasets(
        train_set, test_set, train_transformation, test_transformation), task_labels


def _get_mini_imagenet_dataset(path, train_transformation, test_transformation):
    """ Create from ImageNet. """
    task_labels = None
    train_set = MiniImageNetDataset(path, split='train')
    test_set = MiniImageNetDataset(path, split='test')

    tr_ret, test_ret = train_test_transformation_datasets(
        train_set, test_set, train_transformation, test_transformation)
    return tr_ret, test_ret, task_labels


def get_split_mini_imagenet(root_path, tasks=None, nb_tasks=20):
    import pickle
    import numpy as np

    if tasks is None:
        tasks = [i + 1 for i in range(nb_tasks)]
    if type(tasks) is int:
        tasks = [tasks]

    with open(f"{root_path}/miniImageNet.pkl", "rb") as f:
        dataset = pickle.load(f)

    task_labels = []
    with open(f"{root_path}/miniimagenet_split_{nb_tasks}.txt", "r") as f:
        counter = 1
        while True:
            line = f.readline()
            if not line:
                break
            if counter in tasks:
                task_labels.extend([int(e) for e in line.rstrip().split(" ")])
            counter += 1

    train_x, test_x = [], []
    train_y, test_y = [], []

    for i in range(0, len(dataset["labels"]), 600):
        train_x.extend(dataset["data"][i:i + 500])
        test_x.extend(dataset["data"][i + 500:i + 600])
        train_y.extend(dataset["labels"][i:i + 500])
        test_y.extend(dataset["labels"][i + 500:i + 600])

    train_x, test_x = np.array(train_x), np.array(test_x)
    train_y, test_y = np.array(train_y), np.array(test_y)

    return XYDataset(train_x, train_y), XYDataset(test_x, test_y), task_labels


class XYDataset(Dataset):
    """ Template Dataset with Labels """

    def __init__(self, x, y, **kwargs):
        self.x, self.targets = x, y

        for name, value in kwargs.items():
            setattr(self, name, value)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.targets[idx]

        return x, y


__all__ = [
    'SplitMiniImageNet'
]
