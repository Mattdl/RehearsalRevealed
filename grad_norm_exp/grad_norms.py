import torch
import cole as cl
import numpy as np
import argparse
import os

cl.set_data_path("./data")
device = "cuda"
_BASE_PATH = ".."


def calc_full_grad_norm(loaders, model):
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    n_samples = 0
    opt.zero_grad()
    for loader in loaders:
        for (x, y) in loader:
            x, y = x.to(device), y.to(device)
            n_samples += len(y)
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y, reduction='sum')
            loss.backward()
    grad = torch.cat([p.grad.flatten() for p in model.parameters()])
    grad.div_(n_samples)
    return grad.norm()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=1)
    parser.add_argument('--data', type=str, default="mnist")
    args = parser.parse_args()

    if args.data == "mnist":
        model = cl.MLP().to(device)
        dataset = cl.get_split_mnist((1, 2, 3, 4, 5))
        dataset_t1 = cl.get_split_mnist((1, ))
        buf_size = 50
        epochs = 1
    elif args.data == "cifar":
        model = cl.get_resnet18().to(device)
        dataset = cl.get_split_cifar10((1, 2, 3, 4, 5))
        dataset_t1 = cl.get_split_cifar10((1, ))
        buf_size = 100
        epochs = 1
    elif args.data == "min":
        model = cl.get_resnet18(100, (3, 84, 84)).to(device)
        dataset = cl.get_split_mini_imagenet((1, 2, 3, 4, 5))
        dataset_t1 = cl.get_split_mini_imagenet(1)
        buf_size = 100
        epochs = 10
    else:
        raise ValueError("Data unknown")

    loaders = cl.CLDataLoader(dataset.train, bs=10, task_size=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    buffer = cl.Buffer(0, sampler="balanced")
    results = [[] for _ in range(6)]

    for i in range(args.iter):
        for t, loader in enumerate(loaders):
            buffer.size = (t + 1) * buf_size
            for e in range(epochs):
                for data, target in loader:
                    buffer.sample((data, target))
                    buf_data, buf_target = buffer.retrieve((data, target), size=10)
                    if buf_data is not None:
                        data, target = torch.cat([data, buf_data]), torch.cat([target, buf_target])
                    data, target = data.to(device), target.to(device)
                    cl.step(model, optimizer, data, target)

            if t in [0, 1, 4]:
                idx = 2 if t == 4 else t
                test_loader = cl.CLDataLoader(dataset_t1.test, bs=64)
                buffer_loader = cl.CLDataLoader([buffer], bs=50, shuffle=False, task_size=100)

                full_grad = calc_full_grad_norm(test_loader, model).item()
                buffer_grad = calc_full_grad_norm(buffer_loader, model).item()

                results[idx].append(full_grad)
                results[3 + idx].append(buffer_grad)

    results = np.array(results)
    rand_idx = np.random.randint(0, 10000)

    if not os.path.exists(f"{_BASE_PATH}/results/{args.data}"):
        os.makedirs(f"{_BASE_PATH}/results/{args.data}")

    np.save(f'{_BASE_PATH}/results/{args.data}/grad_norms_{rand_idx}.npy', results)


if __name__ == '__main__':
    main()
