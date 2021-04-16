from abc import ABC

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class XYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None, **kwargs):
        super(XYDataset, self).__init__()
        self.x, self.y = x, y
        self.transform = transform

        for name, value in kwargs.items():
            setattr(self, name, value)

    def get_labels(self):
        return set(self.y)

    def __len__(self):
        return len(self.x)

    # TODO: multiple item getter reverse dimensions for some reason
    def __getitem__(self, item):
        if self.transform is not None:
            x = self.transform(self.x[item])
        else:
            x = self.x[item]
        return x, self.y[item]


class SizedSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, size, batch_size=10, shuffle=True):
        super(SizedSampler, self).__init__(data_source)
        self.bs = batch_size
        self.size = size if len(data_source) > size else len(data_source)
        self.shuffle = shuffle

        if shuffle:
            self.order = torch.randperm(len(data_source)).tolist()[:self.size]
        else:
            self.order = list(range(self.size))

    def __iter__(self):
        batch = []
        for idx in self.order:
            batch.append(idx)
            if len(batch) == self.bs:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

        if self.shuffle:
            random.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(self.size // self.bs))


# TODO: move tasks to loader, such that dataset doesn't have to be reloaded when tasks change, mainly for quicker ex.
def make_split_dataset_old(train, test, joint=False, tasks=None, transform=None):
    train_x, train_y = train.data, train.targets
    test_x, test_y = test.data, test.targets

    # Sort all samples based on targets
    out_train = [(x, y) for (x, y) in sorted(zip(train_x, train_y), key=lambda v: v[1])]
    out_test = [(x, y) for (x, y) in sorted(zip(test_x, test_y), key=lambda v: v[1])]

    # Create tensor of samples and targets
    train_x, train_y = [np.stack([elem[i] for elem in out_train]) for i in [0, 1]]
    test_x, test_y = [np.stack([elem[i] for elem in out_test]) for i in [0, 1]]

    # Get max idx of each target label
    train_idx = [((train_y + i) % 10).argmax() for i in range(10)]
    train_idx = sorted(train_idx) + [len(train_x)]

    test_idx = [((test_y + i) % 10).argmax() for i in range(10)]
    test_idx = sorted(test_idx) + [len(test_x)]

    labels_per_task = 2
    train_ds, test_ds = [], []
    for i in tasks:
        task_st_label = (i - 1) * 2
        tr_s, tr_e = train_idx[task_st_label], train_idx[task_st_label + labels_per_task]
        te_s, te_e = test_idx[task_st_label], test_idx[task_st_label + labels_per_task]

        train_ds += [(train_x[tr_s:tr_e], train_y[tr_s:tr_e])]
        test_ds += [(test_x[te_s:te_e], test_y[te_s:te_e])]

    if joint:
        train_ds = [(np.concatenate([task_ds[0] for task_ds in train_ds]),
                     np.concatenate([task_ds[1] for task_ds in train_ds]))]
        test_ds = [(np.concatenate([task_ds[0] for task_ds in test_ds]),
                    np.concatenate([task_ds[1] for task_ds in test_ds]))]

    train_ds, val_ds = make_valid_from_train(train_ds)

    train_ds = [XYDataset(x[0], x[1], transform=transform) for x in train_ds]
    val_ds = [XYDataset(x[0], x[1], transform=transform) for x in val_ds]
    test_ds = [XYDataset(x[0], x[1], transform=transform) for x in test_ds]

    return DataSplit(train_ds, val_ds, test_ds)


def make_split_dataset(train, test, joint=False, tasks=None, transform=None):
    train_x, train_y = np.array(train.data), np.array(train.targets)
    test_x, test_y = np.array(test.data), np.array(test.targets)

    train_ds, test_ds = [], []

    task_labels = [[(t-1)*2, (t-1)*2 + 1] for t in tasks]
    if joint:
        task_labels = [[label for task in task_labels for label in task]]

    for labels in task_labels:
        train_label_idx = [y in labels for y in train_y]
        test_label_idx = [y in labels for y in test_y]
        train_ds.append((train_x[train_label_idx], train_y[train_label_idx]))
        test_ds.append((test_x[test_label_idx], test_y[test_label_idx]))

    train_ds, val_ds = make_valid_from_train(train_ds)

    train_ds = [XYDataset(x[0], x[1], transform=transform) for x in train_ds]
    val_ds = [XYDataset(x[0], x[1], transform=transform) for x in val_ds]
    test_ds = [XYDataset(x[0], x[1], transform=transform) for x in test_ds]

    return DataSplit(train_ds, val_ds, test_ds)


def make_split_label_set(train, test, label, transform):
    train_x, train_y = np.array(train.data), np.array(train.targets)
    test_x, test_y = np.array(test.data), np.array(test.targets)

    train_label_idx = np.where(train_y == label)
    test_label_idx = np.where(test_y == label)

    train_ds = (train_x[train_label_idx], train_y[train_label_idx])
    train_ds, val_ds = make_valid_from_train([train_ds])
    train_ds, val_ds = train_ds[0], val_ds[0]
    test_ds = (test_x[test_label_idx], test_y[test_label_idx])

    train_ds = [XYDataset(train_ds[0], train_ds[1], transform=transform)]
    val_ds = [XYDataset(val_ds[0], val_ds[1], transform=transform)]
    test_ds = [XYDataset(test_ds[0], test_ds[1], transform=transform)]

    return DataSplit(train_ds, val_ds, test_ds)


class DataSplit:
    def __init__(self, train_ds, val_ds, test_ds):
        self.train = train_ds
        self.validation = val_ds
        self.test = test_ds


def make_valid_from_train(dataset, cut=0.95):
    tr_ds, val_ds = [], []
    for task_ds in dataset:
        x_t, y_t = task_ds

        # shuffle before splitting
        perm = torch.randperm(len(x_t))
        x_t, y_t = x_t[perm], y_t[perm]

        split = int(len(x_t) * cut)
        x_tr, y_tr = x_t[:split], y_t[:split]
        x_val, y_val = x_t[split:], y_t[split:]

        tr_ds += [(x_tr, y_tr)]
        val_ds += [(x_val, y_val)]

    return tr_ds, val_ds


def test_dataset(model, loader, device='cpu', loss_func=None):
    """
    Return the loss and accuracy on a single dataset (which is provided through a loader)
    Interference is based on argmax of outputs.
    """
    loss, correct, length = 0, 0, 0

    if loss_func is None:
        loss_func = loss_wrapper("CE", reduction='sum')

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += loss_func(output, target, model)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            length += len(target)

    return loss / length, correct / length


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module, ABC):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.bn1 = nn.Sequential()
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False)
        self.bn2 = nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                # nn.BatchNorm2d(self.expansion * planes, track_running_stats=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module, ABC):
    def __init__(self, block, num_blocks, num_classes, nf, input_size):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        self.conv1 = conv3x3(input_size[0], nf * 1)
        # self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False)
        self.bn1 = nn.Sequential()
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        last_hid = nf * 8 * block.expansion if input_size[1] in [8, 16, 21, 32, 42] else 640
        self.linear = nn.Linear(last_hid, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        # pre_bn = self.conv1(x.view(bsz, 3, 32, 32))
        # post_bn = self.bn1(pre_bn, 1 if is_real else 0)
        # out = F.relu(post_bn)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out


def loss_wrapper(loss='CE', **kwargs):
    """
    Create universal wrapper around loss functions, such that they can all be called with the same arguments,
    (data, target, model, **kwargs) which is useful in other functions that use losses.
    :param loss: "CE" for cross entropy. Optional param "reduction". "hinge" for hinge loss (c * l2 + hinge). Optional
    parameters are "margin", "c" and "reduction".
    :return:
    """

    if loss == 'CE':
        reduction = kwargs["reduction"] if "reduction" in kwargs else "mean"

        def ce_loss(data, target, model):
            return F.cross_entropy(data, target, reduction=reduction)
        return ce_loss

    elif loss == "hinge":
        margin = kwargs["margin"] if "margin" in kwargs else 1.0
        c = kwargs["c"] if "c" in kwargs else 0.001
        reduction = kwargs["reduction"] if "reduction" in kwargs else "mean"

        def hinge(data, target, model):
            return torch.add(c * l2_loss(model), hinge_loss(data, target, margin=margin, reduction=reduction))
        return hinge


def hinge_loss(output, target, margin=1.0, reduction='mean'):
    """
    Calculates multi label hinge loss as sum(max(0, margin + w^T x - w_i^t x))
    :return: mean loss
    """
    loss = torch.tensor(0.0, requires_grad=True)
    for x, y in zip(output, target):
        s_loss = margin + (x - x[y])
        s_loss[s_loss < 0] = 0
        loss = torch.add(loss, torch.sum(s_loss) - margin)
        # Max version of hinge loss. Theoreticaly beter, not so in practice
        # s_loss = margin + x - x[y]
        # s_loss[s_loss < 0] = 0
        # if torch.argmax(s_loss) == y:
        #     continue
        # else:
        #     If this is equal to the correct label, its derivative will be 0
        #     loss = torch.add(loss, torch.pow(torch.max(s_loss), 2))

    loss = loss / len(target) if reduction == 'mean' else loss
    return loss


def l2_loss(model):
    """
    Calculates l2 norm of weights. Can be used as loss.
    """
    return torch.sum(torch.stack([torch.norm(p) for p in model.parameters()]))