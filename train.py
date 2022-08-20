import os
import argparse

from libs.utils import *
from libs.train import *
from libs.train_cahr import *
from omegaconf import OmegaConf


def main(args):

    # loads configs
    configs = OmegaConf.load(args.config_file)

    # initialize environments
    init_environment(configs.seed)
    exp_dir = os.path.join(args.exp_root, configs.exp)

    if args.trainer == 'basic':
        # loads dataloder for training and validation
        train_loader = get_dataloader('train', args.train_dir, configs.loader)
        val_loader   = get_dataloader('val',   args.val_dir,   configs.loader)

        # initialize trainer
        trainer = BCITrainer(configs, exp_dir, args.resume_ckpt)

        # training model
        trainer.forward(train_loader, val_loader)

    elif args.trainer == 'cahr':
        # loads dataloder for training and validation
        train_loader = get_cahr_dataloader('train', args.train_dir, configs.loader)
        # val_loader   = get_cahr_dataloader('val',   args.train_dir, configs.loader)

        for data in train_loader:
            pass

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training for BCI Dataset')
    parser.add_argument('--train_dir',   type=str, help='dir path of training data')
    parser.add_argument('--val_dir',     type=str, help='dir path of validation data')
    parser.add_argument('--exp_root',    type=str, help='root dir of experiment')
    parser.add_argument('--config_file', type=str, help='yaml path of configs')
    parser.add_argument('--resume_ckpt', type=str, help='checkpoint path for resuming')
    parser.add_argument('--trainer',     type=str, help='trainer type, basic or cahr', default='basic')
    args = parser.parse_args()

    check_train_args(args)
    main(args)
