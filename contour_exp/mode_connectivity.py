import cole as cl
import numpy as np
import argparse
import tqdm
import json
import torch
import os
import pickle

cl.set_data_path('../data')


def test_xy(x, y, plane, model, loaders, device="cpu", loss_func=None):
    params = plane.xy_to_weights(x, y)
    cl.assign_parameters(model, params)
    return cl.test(model, loaders, avg=False, device=device, loss_func=loss_func)


def main():
    device = 'cuda'
    base_path = '..'

    seed = torch.randint(1000000, (1, ))
    # seed = 1997
    print(f"Seed: {seed}")
    torch.manual_seed(seed)

    parser = argparse.ArgumentParser()

    parser.add_argument('models', nargs=3,
                        help="Three models that define the plane, stored at ./models/{data}/")
    parser.add_argument('--data', default='mnist',
                        choices=['mnist', 'cifar', 'min'])
    parser.add_argument('--tasks', default="12", help="Which tasks to use")
    parser.add_argument('--size', type=int, default=0,
                        help="Size of (task) test set, 0 is full set")
    parser.add_argument('--buffer', nargs="*", default=None,
                        help="Name of buffers pickle objects if buffers are to be used. Will be last in test array.")
    parser.add_argument('--path', nargs="*", default=None,
                        help="If paths are stored for a model, calculate projections of the paths in arg")
    parser.add_argument('-g', '--grid', default=10, type=int,
                        help='Size of the hyperplane grid (g x g)')
    parser.add_argument('--start', type=float, default=-0.5,
                        help="Relative coordinate of start grid. First x = start * (w2 - w1)")
    parser.add_argument('--width', type=float, default=2.0,
                        help="Relative width of the grid")
    parser.add_argument('--save', default=None, help="Save (and name of file)")
    parser.add_argument('--threads', default=1, type=int,
                        help="Nb of threads, 1 will be serial, can fail on GPU")

    args = parser.parse_args()
    args.tasks = [int(i) for i in args.tasks]

    models = [cl.build_model(f"{base_path}/models/{args.data}/{m}.pt", args.data) for m in args.models]
    plane = cl.WeightPlane(*(cl.flatten_parameters(m) for m in models))
    loss_func = cl.loss_wrapper("CE", reduction='sum')

    buffers = []
    if args.buffer is not None:
        for buf_name in args.buffer:
            with open(f"{base_path}/models/{args.data}/{buf_name}_buffer.pkl", "rb") as f:
                buffers.append(pickle.load(f))

    for path in args.path:
        path_xy = []
        model_path = sorted(os.listdir(f"{base_path}/models/{args.data}/{path}/"), key=lambda n: int(n[6:-3]))
        for model in model_path:
            model = cl.build_model(f"{base_path}/models/{args.data}/{path}/{model}", args.data)
            x, y = plane.project_onto_plane(cl.flatten_parameters(model))
            path_xy.append([x, y])
        np.save(f"{base_path}/models/{args.data}/{path}_path.npy", np.array(path_xy))

    if args.data == 'mnist':
        data = cl.get_split_mnist(tasks=args.tasks)
        inter_model = cl.MLP()
    elif args.data == "cifar":
        data = cl.get_split_cifar10(tasks=args.tasks)
        inter_model = cl.get_resnet18()
    elif args.data == "min":
        data = cl.get_split_mini_imagenet(tasks=args.tasks)
        inter_model = cl.get_resnet18(100, (3, 84, 84))
    else:
        raise ValueError("Unknown data")

    x_co = np.linspace(args.start, args.start + args.width, args.grid)
    y_co = np.linspace(args.start, args.start + args.width, args.grid)

    inter_model.to(device)

    test_loader = cl.CLDataLoader([*data.test, *buffers], bs=256, shuffle=True, task_size=args.size)
    result_mat = np.zeros((len(x_co), len(y_co), 2, len(test_loader)))

    if args.threads > 1:
        # Only do imports here otherwise script fails if package isn't installed.
        from joblib import Parallel, delayed
        from itertools import product

        print(f"Total jobs: {len(x_co) * len(y_co)}")
        results = Parallel(n_jobs=args.threads, verbose=5)(
            delayed(test_xy)(x, y, plane, inter_model, test_loader, device, loss_func)
            for x, y in product(x_co, y_co))

        for i, row in enumerate(results):
            result_mat[i // args.grid, i % args.grid] = row

    else:
        progress_bar = tqdm.tqdm(total=len(x_co) * len(y_co))
        for x_i, x in enumerate(x_co):
            for y_i, y in enumerate(y_co):
                result_mat[x_i, y_i] = test_xy(x, y, plane, inter_model, test_loader, device, loss_func)
                progress_bar.update(1)

    if args.save:
        if not os.path.exists(f"../results/{args.data}"):
            os.makedirs(f"../results/{args.data}")

        model_coordinates = plane.weights_to_xy()
        rel_path = f'{base_path}/results/{args.data}/{args.save}_{args.grid}_{args.size}'
        np.save(f'{rel_path}_models.npy', model_coordinates)
        np.save(f'{rel_path}.npy', result_mat)

        with open(f"{rel_path}.json", 'w') as f:
            json.dump(vars(args), f, indent=2)


if __name__ == '__main__':
    main()
