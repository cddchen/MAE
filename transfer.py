import argparse
import os

import torchvision.utils
import yaml
import torch
from datetime import datetime
from datasets import get_dataset
from mae import get_model
from latent2class_model import Latent2Class
from utils import set_debug, get_optimizer, get_scheduler


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=None, help='The folder that has all the datasets.')
    parser.add_argument('--config', '-c', type=str, default='./configs/imagenet.yaml',
                        help="Yaml config file. Don't forget to 'pip install pyyaml' first")
    parser.add_argument('--output_dir', type=str, default='../mae_out/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--subset_size', type=int, default=2, help='Used along with the --debug command, \
                        run the whole program with only a few samples and find debugs, or the best batch size')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--resume', type=str,
                        default='checkpoints/ep199.pth',
                        help='path of checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        for key, value in yaml.load(f, Loader=yaml.FullLoader).items():
            vars(args)[key] = value

    if args.data_path is not None:
        args.dataset.root = args.data_path

    args.output_dir = os.path.join(os.path.abspath(args.output_dir),
                                   *args.config.rstrip('.yaml').split('/')[1:]) + \
                      '-' + datetime.now().strftime('%m%d%H%M%S')

    os.makedirs(args.output_dir, exist_ok=False)
    print(f'Outputs will be saved to {args.output_dir}')

    return args


def main(args):
    print(f'Found {torch.cuda.device_count()} gpu(s)')
    train_set = get_dataset(
        split='train',
        image_size=args.image_size,
        **args.dataset
    )
    test_set = get_dataset(
        split='val',
        image_size=args.image_size,
        **args.dataset
    )

    # args, train_set, test_set = set_debug(args, train_set, test_set)

    train_loader = torch.utils.data.dataloader.DataLoader(
        dataset=train_set,
        num_workers=args.num_workers,
        **args.train_loader
    )

    test_loader = torch.utils.data.dataloader.DataLoader(
        dataset=test_set,
        num_workers=5,
        **args.test_loader
    )

    model = get_model(image_size=args.image_size, **args.model).to(args.device)
    latent2class = Latent2Class().to(args.device)
    model = torch.nn.parallel.DataParallel(model)
    latent2class = torch.nn.parallel.DataParallel(latent2class)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(latent2class.parameters(), lr=1e-3)

    optimizer = get_optimizer(model, **args.optimizer)
    scheduler = get_scheduler(**args.scheduler)

    scheduler.set_optimizer(optimizer)

    args.param_num = sum(p.numel() for p in latent2class.parameters() if p.requires_grad)
    print(f'Param num: {args.param_num}')

    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict)
        print('Load state dict success.')


    for epoch in range(args.epochs):
        model.train()
        latent2class.train()
        acc_list = []

        for idx, (images, labels) in enumerate(train_loader):
            # lr = scheduler.step(epoch, args.epochs, idx, len(train_loader))
            latent2class.zero_grad()
            model.zero_grad()

            images, labels = images.to(args.device), labels.to(args.device)

            latent = model(images, transfer=True)
            prob = latent2class(latent.detach())
            loss = criterion(prob, labels)
            loss.backward()
            opt.step()
            # optimizer.step()
            acc = torch.sum(torch.argmax(prob, dim=1) == labels)
            acc_item = acc.item() / images.shape[0]
            acc_list.append(acc_item)
            print(f'\riter: {idx}/{len(train_loader)}, loss: {loss.item():.2f}, acc: {acc_item * 100:.2f}%', end='')
        avg_loss = sum(acc_list) / len(acc_list)
        print(f'\tEpoch {epoch} Avg Accuracy: {avg_loss * 100:.2f}%', end='\n')

        model_to_save = latent2class
        # model_to_save = model_to_save.module if hasattr(model_to_save, "module") else model_to_save
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, f'latent2cls_ep{epoch}.pth'))
        # model_to_save = model
        # torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, f'ep{epoch}.pth'))

    print(f'Output has been saved to {args.output_dir}')


if __name__ == '__main__':
    main(get_args())