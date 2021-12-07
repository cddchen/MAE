import argparse
import os

import torchvision.utils
import yaml
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import get_dataset
from mae import get_model
from utils import set_debug, get_optimizer, get_scheduler


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default=None, help='The folder that has all the datasets.')
    parser.add_argument('--config', '-c', type=str, default='./configs/eval_imagenet.yaml',
                        help="Yaml config file. Don't forget to 'pip install pyyaml' first")
    parser.add_argument('--output_dir', type=str, default='../mae_out/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--subset_size', type=int, default=2, help='Used along with the --debug command, \
                        run the whole program with only a few samples and find debugs, or the best batch size')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--resume', type=str, default='/media/yz/data_drive0/21_chendong/MAE/checkpoints/ep0.pth', help='path of checkpoint')
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
        num_workers=args.num_workers,
        **args.test_loader
    )

    model = get_model(image_size=args.image_size, **args.model).to(args.device)
    model = torch.nn.parallel.DataParallel(model)
    if args.resume is not None:
        state_dict = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(state_dict)
        print('Load state dict success.')

    model.eval()
    with torch.no_grad():
        images, _ = next(iter(train_loader))
        images = images.to(args.device)
        orig_images = torchvision.utils.make_grid(((images + 1) / 2.).cpu(), nrow=5)
        out = model(images, viz=True)
        recon = out['recon']
        recon_image = torchvision.utils.make_grid(((recon + 1) / 2.).cpu(), nrow=5)

        fig, axs = plt.subplots(6, 1, figsize=(64,64))
        fig.subplots_adjust(wspace=0, hspace=0.1)
        axs[0].imshow(orig_images.permute(1, 2, 0))
        axs[0].set_ylabel('original', fontsize=50)
        for i in range(2, 7):
            state_dict = torch.load(f'/media/yz/data_drive0/21_chendong/MAE/checkpoints/ep{40 * (i - 1) - 1}.pth', map_location='cpu')
            model.load_state_dict(state_dict)
            out = model(images, viz=True)
            recon = out['recon']
            recon_image = torchvision.utils.make_grid(((recon + 1) / 2.).cpu(), nrow=5)
            axs[i - 1].imshow(recon_image.permute(1, 2, 0))
            axs[i - 1].set_ylabel(f'ep_{40 * (i - 1) - 1}', fontsize=50)
        # plt.show()
        plt.savefig('result.png')

if __name__  == '__main__':
    main(get_args())