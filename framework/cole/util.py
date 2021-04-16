import numpy as np
import random
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.autograd as autograd
from cole.core import *


def build_model(path, data='mnist', device='cpu'):
    """
    Loads and returns model at a given path, and sends to device
    """
    if data == "mnist":
        model = MLP()
    elif data == "cifar":
        model = get_resnet18()
    elif data == "min":
        model = get_resnet18(100, (3, 84, 84))
    else:
        raise ValueError(f"Unknown model, got {data}")

    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


def laplace_matrix(a):
    """
    Calculate the Laplace Matrix of a given matrix a.
    :param a: 2D numpy array
    :return: 2D numpy array with result
    """
    n, m = a.shape
    result = np.zeros((n-2, m-2))
    for i in range(1, n-1):
        for j in range(1, m-1):
            result[i-1, j-1] = a[i-1, j] + a[i+1, j] + a[i, j+1] + a[i, j-1] - 4*a[i, j]
    return result


def hessian(gradient, model):
    """
    Calculates Hessian matrix to the model parameters, based of the passed in gradient.
    :param gradient: Gradient tensor. Has to be calculated with create_graph=True
    :param model: model with parameters()
    :return: Torch tensor NxN, with N model parameters
    """
    hess = torch.empty((gradient.numel(), gradient.numel()))
    for i, grad_element in enumerate(gradient):
        hess_row = autograd.grad(grad_element, model.parameters(), retain_graph=True)
        hess_row = torch.cat([h.view(-1) for h in hess_row])
        hess[i] = hess_row
    return hess


def flatten_parameters(model):
    """
    Returns the model weights as a 1D tensor
    :return: 1D torch tensor with size N, with N the model parameters.
    """
    with torch.no_grad():
        flat = torch.cat([
            p.view(-1) for p in model.parameters()
        ])
    return flat


# TODO: merge with flatten parameters, or find better name. Test speed gain if grad not kept
def flatten_parameters_wg(model):
    """
    Flattens parameters of a model but retains the gradient
    :return: 1D torch tensor with size N, with N the model paramters
    """
    return torch.cat([p.view(-1) for p in model.parameters()])


def assign_parameters(model, params):
    """
    Assigns a flat parameter tensor back to a model. Batch norm layers weights are skipped.
    :param model: model layout, params will be overwritten.
    :param params: 1D tensor
    :return model. This is the same model as the passed one, since no model copy is made.
    """
    state_dict = model.state_dict(keep_vars=False)
    with torch.no_grad():
        total_count = 0
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            param_count = state_dict[param].numel()
            param_size = state_dict[param].shape
            state_dict[param] = params[total_count:total_count+param_count].view(param_size).clone()
            total_count += param_count
        model.load_state_dict(state_dict)
    return model


def create_buffer(data: DataSplit, size, sampler='balanced', shuffle=True):
    """
    :param data: dataset to use (first task will be used first)
    :param size: size of dataset
    :param sampler: sampler method to use for buffer
    :param shuffle: shuffle loader before using data
    :return: buffer
    """
    buffer = Buffer(size, sampler=sampler)
    loaders = CLDataLoader(data.train, bs=1, shuffle=shuffle)

    for loader in loaders:
        for data, target in loader:
            if len(buffer) == size and sampler == "first_in":
                break
            buffer.sample((data, target))
        else:
            continue
        break

    return buffer


class WeightPlane:
    def __init__(self, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor):
        """
        ! WARNING: u and v tend to have numerical errors.
        Constructs a 2D plane spanned by w2 - w1 and w3 - w1. Parameters should be 1D.
        """
        self.w1, self.w2, self.w3 = w1, w2, w3

        self.u = w2 - w1
        self.u_norm = self.u.norm()
        self.u /= self.u_norm

        self.v = w3 - w1
        self.v -= torch.dot(self.u, self.v) * self.u
        self.v_norm = self.v.norm()
        self.v /= self.v_norm

    def xy_to_weights(self, x, y):
        """
        Transform 2D coordinates in the plane back to original coordinates.
        """
        return self.w1 + x * self.u_norm * self.u + y * self.v_norm * self.v

    def weights_to_xy(self):
        """
        :return: numpy 2D array with the coordinates of w1, w2 and w3 in the 2D plane
        """
        non_orth_v = (self.w3 - self.w1)
        w3_x = torch.dot(self.u, non_orth_v) / self.u_norm
        model_weights = [[0, 0], [1, 0], [w3_x.item(), 1]]
        return np.array(model_weights)

    def project_onto_plane(self, weights: torch.Tensor):
        """
        :return: xy coordinates of the projection onto the plane span by u and v.
        """
        x = torch.dot(self.u, weights - self.w1) / self.u_norm
        y = torch.dot(self.v, weights - self.w1) / self.v_norm
        return x.item(), y.item()

    def inter_model_dist(self):
        """
        :return: l2-distances between models used to construct plane. (w1 - w2, w2 - w3, w1 - w3)
        """
        dist_1 = torch.dist(self.w1, self.w2)
        dist_2 = torch.dist(self.w2, self.w3)
        dist_3 = torch.dist(self.w3, self.w1)

        return dist_1, dist_2, dist_3
