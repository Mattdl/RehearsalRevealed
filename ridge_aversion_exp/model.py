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
These are the ResNet and MLP models used for the MNIST,CIFAR and Mini-Imagenet experiments.
"""

import math
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d


class MyMLP(nn.Module):

    def __init__(self, num_classes=10, input_size=28 * 28, hidden_size=512):
        super(MyMLP, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._input_size = input_size
        print(f"[MODEL] MyMLP, 2 layers, featsize={hidden_size}, outputs={num_classes}, inputsize={input_size}")

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.size(0), self._input_size)
        x = self.features(x)
        x = self.classifier(x)
        return x


def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockStable(nn.Module):
    """ STABLE-SGD VERSION"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_prob=0):
        super(BasicBlockStable, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
            )
        self.IC1 = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.Dropout(p=drop_prob)
        )

        self.IC2 = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.Dropout(p=drop_prob)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = relu(out)
        out = self.IC1(out)

        out += self.shortcut(x)
        out = relu(out)
        out = self.IC2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf, bias, drop_prob, input_size=(3, 32, 32)):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.input_size = input_size

        if input_size == (3, 32, 32):  # CIFAR
            factor = 8
        elif input_size == (3, 84, 84):  # Mini Imgnet
            factor = 8 * 4
        else:
            raise NotImplementedError()

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, drop_prob=drop_prob)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, drop_prob=drop_prob)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, drop_prob=drop_prob)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, drop_prob=drop_prob)
        print("BIAS IS", bias)
        self.linear = nn.Linear(nf * factor * block.expansion, num_classes, bias=bias)

    def _make_layer(self, block, planes, num_blocks, stride, drop_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, drop_prob))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # torch.Size([10, 160, 2, 2]) # torch.Size([10, 640])
        out = self.linear(out)  # 160
        return out


def ResNet18(input_size, nclasses, nf=20, bias=True, drop_prob=0):
    print(f"[MODEL] Resnet18, outputs={nclasses}, feats={nf}, bias={bias}, DROP_PROB={drop_prob}")
    return ResNet(BasicBlockStable, [2, 2, 2, 2], nclasses, nf, bias, drop_prob, input_size)


if __name__ == "__main__":

    kwargs = {'num_classes': 10}
    model = MyMLP(**kwargs)

    for name, module in model.named_parameters():
        print(name)
