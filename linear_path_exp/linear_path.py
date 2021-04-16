import torch
import cole as cl
import numpy as np
import argparse
import csv
from copy import deepcopy
import os


cl.set_data_path("./data")
device = "cuda"
_BASE_PATH = ".."


def calc_loss_path(first, second, loaders, alpha=10, data="mnist", path=None):

    if path is None:
        print("[WARNING] Not saving path intermediate losses")
    losses = []

    first_flat = cl.flatten_parameters(first)
    second_flat = cl.flatten_parameters(second)
    alphas = torch.linspace(0, 1, alpha)

    if data == "mnist":
        inter_model = cl.MLP()
    elif data == "cifar":
        inter_model = cl.get_resnet18()
    elif data == "min":
        inter_model = cl.get_resnet18(100, (3, 84, 84))
    else:
        raise ValueError(f"Data {data} unknown")

    inter_model.to(device)
    for a in alphas:
        new_param = (1 - a) * first_flat + a * second_flat
        cl.assign_parameters(inter_model, new_param)
        acc, loss = cl.test(inter_model, loaders, device=device)
        losses.append(loss)

    rand_id = np.random.randint(0, 1_000_000_000)

    if not os.path.isfile(path):
        with open(path, 'w') as f:
            header = "idx,"
            for i in range(len(losses)):
                header += f"{i},"
            f.write(header[:-1])
            f.write("\n")

    with open(path, 'a+') as f:
        writer = csv.writer(f)
        losses.insert(0, rand_id)
        writer.writerow(losses)
    return rand_id


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="mnist")
    parser.add_argument('--f_idx', type=str, default="m",
                        help="File index for saving results (to guarantee no clashes occur)")
    args = parser.parse_args()

    path = f"{_BASE_PATH}/results/{args.data}"
    if not os.path.exists(path):
        os.makedirs(path)

    path_path = f"{path}/linear_path_raw_{args.f_idx}.csv"

    if args.data == "mnist":
        model = cl.MLP().to(device)
        dataset = cl.get_split_mnist((1, 2, 3, 4, 5))
        dataset_t1 = cl.get_split_mnist(1)
        dataset_t12 = cl.get_split_mnist((1, 2))
        buf_size = 50
        epochs = 1
    elif args.data == "cifar":
        model = cl.get_resnet18().to(device)
        dataset = cl.get_split_cifar10((1, 2, 3, 4, 5))
        dataset_t1 = cl.get_split_cifar10(1)
        dataset_t12 = cl.get_split_cifar10((1, 2))
        buf_size = 100
        epochs = 1
    elif args.data == "min":
        model = cl.get_resnet18(100, (3, 84, 84)).to(device)
        dataset = cl.get_split_mini_imagenet((1, 2, 3, 4, 5))
        dataset_t1 = cl.get_split_mini_imagenet(1)
        dataset_t12 = cl.get_split_mini_imagenet((1, 2))
        buf_size = 100
        epochs = 10
    else:
        raise ValueError

    path_idx = []
    loaders = cl.CLDataLoader(dataset.train, bs=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Create two DIFFERENT buffers (shuffle = True)
    buffers = [cl.create_buffer(dataset_t1, size=buf_size, sampler="balanced", shuffle=True) for _ in range(2)]
    orig_buffers = deepcopy(buffers)
    buffer_loaders = [cl.CLDataLoader([buffer], bs=10) for buffer in orig_buffers]

    # First train task 1, create two buffers
    for e_idx in range(epochs):
        for data, target in loaders[0]:
            data, target = data.to(device), target.to(device)
            cl.step(model, optimizer, data, target)

    print("Task 1 done")

    # Next task, two models
    models = [model, deepcopy(model)]
    original_t1_model = deepcopy(model)
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.01) for m in models]

    # train task 2
    for model, buffer, optimizer in zip(models, buffers, optimizers):
        buffer.size = 2 * buf_size
        for e_idx in range(epochs):
            for data, target in loaders[1]:
                buffer.sample((data, target))
                buf_data, buf_target = buffer.retrieve((data, target), size=10)
                if buf_data is not None:
                    data, target = torch.cat([data, buf_data]), torch.cat([target, buf_target])
                data, target = data.to(device), target.to(device)
                cl.step(model, optimizer, data, target)

    print("Task 2 done")

    # Create loaders
    test_loader_task_12 = cl.CLDataLoader(dataset_t12.test, bs=64)
    train_loader_task_1 = cl.CLDataLoader(dataset_t1.test, bs=64)
    test_loader_task_1 = cl.CLDataLoader(dataset_t1.train, bs=64, task_size=2500)

    # Calc paths origin to w2 / w2'
    path_idx.append(calc_loss_path(original_t1_model, models[0], test_loader_task_1, data=args.data, path=path_path))
    path_idx.append(calc_loss_path(original_t1_model, models[0], train_loader_task_1, data=args.data, path=path_path))
    path_idx.append(calc_loss_path(original_t1_model, models[0], buffer_loaders[0], data=args.data, path=path_path))
    path_idx.append(calc_loss_path(original_t1_model, models[1], buffer_loaders[1], data=args.data, path=path_path))

    # Calc paths from w2 to w2'
    path_idx.append(calc_loss_path(*models, test_loader_task_1, data=args.data, path=path_path))
    path_idx.append(calc_loss_path(*models, test_loader_task_12, data=args.data, path=path_path))
    path_idx.append(calc_loss_path(*models, buffer_loaders[0], data=args.data, path=path_path))
    path_idx.append(calc_loss_path(*models, buffer_loaders[1], data=args.data, path=path_path))

    # train on task 3, 4, 5
    task_idx = 2
    for loader in loaders[2:]:
        for model, buffer, optim in zip(models, buffers, optimizers):
            buffer.size = (task_idx + 1) * buf_size
            for e_idx in range(epochs):
                for data, target in loader:
                    buffer.sample((data, target))
                    buf_data, buf_target = buffer.retrieve((data, target), size=10)
                    if buf_data is not None:
                        data, target = torch.cat([data, buf_data]), torch.cat([target, buf_target])
                    data, target = data.to(device), target.to(device)
                    cl.step(model, optim, data, target)
        task_idx += 1
        print(f"Task {task_idx} done")

    test_loader_full = cl.CLDataLoader(dataset.test, bs=64)

    # Calc paths origin to w5 / w5'
    path_idx.append(calc_loss_path(original_t1_model, models[0], test_loader_task_1, data=args.data, path=path_path))
    path_idx.append(calc_loss_path(original_t1_model, models[0], train_loader_task_1, data=args.data, path=path_path))
    path_idx.append(calc_loss_path(original_t1_model, models[0], buffer_loaders[0], data=args.data, path=path_path))
    path_idx.append(calc_loss_path(original_t1_model, models[1], buffer_loaders[1], data=args.data, path=path_path))

    # Calc paths from w5 to w5'
    path_idx.append(calc_loss_path(*models, test_loader_task_1, data=args.data, path=path_path))
    path_idx.append(calc_loss_path(*models, test_loader_full, data=args.data, path=path_path))
    path_idx.append(calc_loss_path(*models, buffer_loaders[0], data=args.data, path=path_path))
    path_idx.append(calc_loss_path(*models, buffer_loaders[1], data=args.data, path=path_path))

    random_id = np.random.randint(0, 1_000_000)

    # Write file headers if it doesn't exist yet
    if not os.path.isfile(f"{path}/linear_path_idx_{args.f_idx}.csv"):
        with open(f"{path}/linear_path_idx_{args.f_idx}.csv", "w") as f:
            f.write("w1_w2_test_1,w1_w2_train_1,w1_w2_mem_1,w1_w2_mem_2,w2_w2_test_1,w2_w2_test_12,w2_w2_mem_1,"
                    "w2_w2_mem_2,w1_w5_test_1,w1_w5_train_1,w1_w5_mem_1,w1_w5_mem_2,w5_w5_test_1,w5_w5_test_15,"
                    "w5_w5_mem_1,w5_w5_mem_2\n")

    with open(f"{path}/linear_path_idx_{args.f_idx}.csv", 'a+') as f:
        writer = csv.writer(f)
        path_idx.insert(0, random_id)
        writer.writerow(path_idx)


if __name__ == '__main__':
    main()
