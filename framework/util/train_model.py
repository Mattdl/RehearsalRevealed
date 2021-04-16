import torch
import cole as cl
import argparse
import os
import pickle
import json

cl.set_data_path('./data')


def train(args, data, model):

    train_loader = cl.CLDataLoader(data.train, bs=args.bs, shuffle=True)
    # Test on validation set if available (not so for miniIN), else on test set.
    try:
        val_loader = cl.CLDataLoader(data.validation, bs=args.bs, shuffle=False)
    except TypeError:
        val_loader = cl.CLDataLoader(data.test, bs=args.bs, shuffle=False)

    device = torch.device("cpu" if args.no_cuda else "cuda")
    buffer = cl.Buffer(args.buf_size)  # Size will be overwritten if buffer is loaded

    if args.init is not None:
        model.load_state_dict(torch.load(f"../models/{args.data}/{args.init}.pt"))
        if args.buffer:
            if args.buf_name is None:
                buffer_file_name = f"../models/{args.data}/{args.init}_buffer.pkl"
            else:
                buffer_file_name = f"../models/{args.data}/{args.buf_name}.pkl"

            with open(buffer_file_name, 'rb') as f:
                buffer = pickle.load(f)

    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    loss_func = cl.loss_wrapper(args.loss)

    if args.save:
        if not os.path.exists(f"../models/{args.data}"):
            os.makedirs(f"../models/{args.data}")

    if args.save_path:
        os.mkdir(f"../models/{args.data}/{args.save}", 0o750)
        torch.save(model.state_dict(), f"../models/{args.data}/{args.save}/model_0.pt")

    for task, tr_loader in enumerate(train_loader):
        print(f' --- Started training task {task} ---')
        for epoch in range(args.epochs):
            print(f' --- Started epoch {epoch} ---')
            model.train()

            for i, (data, target) in enumerate(tr_loader):
                data, target = data.to(device), target.to(device)

                if args.buffer:
                    buffer.sample((data, target))
                    buf_data, buf_target = buffer.retrieve((data, target), args.bs)
                    if buf_data is not None:
                        buf_data, buf_target = buf_data.to(device), buf_target.to(device)
                        data = torch.cat((data, buf_data))
                        target = torch.cat((target, buf_target))

                cl.step(model, opt, data, target, loss_func)

                if i != 0 and i % args.test == 0:
                    if args.save_path:
                        torch.save(model.state_dict(), f"../models/{args.data}/{args.save}/model_{i}.pt")
                    acc, loss = cl.test(model, val_loader, avg=True, device=device)
                    print(f"\t Acc task {task}, step {i} / {len(tr_loader)}: {acc:.2f}% (Loss: {loss:3f})")

    acc, loss = cl.test(model, val_loader, avg=True, device=device)
    print(f"Final average acc: {acc:.2f}%, (Loss: {loss:3f})")

    if args.save is not None:
        with open(f'../models/{args.data}/{args.save}.json', 'w') as g:
            json.dump(vars(args), g, indent=2)

        torch.save(model.state_dict(), f'../models/{args.data}/{args.save}.pt')
        if args.save_path:
            torch.save(model.state_dict(), f"../models/{args.data}/{args.save}/model_{i}.pt")
        if args.buffer:
            with open(f"../models/{args.data}/{args.save}_buffer.pkl", 'wb+') as f:
                pickle.dump(buffer, f)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for SGD')
    parser.add_argument('--mom', type=float, default=0,
                        help='momentum for SGD')
    parser.add_argument('--bs', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--test', type=int, default=200,
                        help='how many steps before test')
    parser.add_argument('--tasks', type=str, default=None,
                        help='Which tasks should be trained. Used as "123" for task 1, 2 and 3')
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--buffer', action='store_true',
                        help="Use replay buffer")
    parser.add_argument('--buf_name', type=str, default=None,
                        help="Use if buffer to load has different name than the model")
    parser.add_argument('--buf_size', type=int, default=50,
                        help="Size of buffer to be used. Default is 50")
    parser.add_argument('--save', type=str, default=None,
                        help="file name for saving model")
    parser.add_argument('--init', type=str, default=None,
                        help="initial_weights")
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar', 'min'], help="Dataset to train")
    parser.add_argument('--epochs', type=int, default=1, help="Nb of epochs")
    parser.add_argument('--det', action='store_true', help="Run in deterministic mode")
    parser.add_argument('--loss', choices=['CE', 'hinge'], default="CE", help="Loss function for optimizer")
    parser.add_argument('--save_path', action='store_true',
                        help='Store intermediate models at [test] intervals will crash if path (i.e. folder) already'
                             ' exists to prevent accidental overwrites')
    parser.add_argument('--comment', type=str, help="Comment, will be written to json file if model is stored")

    args = parser.parse_args()

    if args.tasks is not None:
        args.tasks = [int(i) for i in args.tasks]

    if not args.no_cuda:
        if torch.cuda.is_available():
            print(f"[INFO] using cuda")
        else:
            args.no_cuda = True

    if args.det:
        torch.manual_seed(1997)

    if args.data == 'mnist':
        data = cl.get_split_mnist(args.tasks, args.joint)
        model = cl.MLP(hid_nodes=400, down_sample=1)
    elif args.data == "cifar":
        data = cl.get_split_cifar10(args.tasks, args.joint)
        model = cl.get_resnet18()
    elif args.data == "min":
        data = cl.get_split_mini_imagenet(args.tasks, nb_tasks=20)
        model = cl.get_resnet18(nb_classes=100, input_size=(3, 84, 84))
    else:
        raise ValueError(f"Data {args.data} not known.")

    train(args, data, model)


if __name__ == '__main__':
    main()
