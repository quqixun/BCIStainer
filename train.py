import os
import argparse

from libs.utils import *
from libs.train import *
from omegaconf import OmegaConf


def main(args):

    # loads configs
    configs = OmegaConf.load(args.config_file)

    # initialize environments
    init_environment(configs.seed)

    # loads dataloder for training and validation
    train_loader = get_dataloader('train', args.train_dir, configs.loader)
    val_loader   = get_dataloader('val',   args.val_dir,   configs.loader)

    # initialize trainer
    exp_dir = os.path.join(args.exp_root, configs.exp)
    trainer = BCITrainer(configs, exp_dir, args.resume_ckpt)

    # training model
    trainer.train_and_val(train_loader, val_loader)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training for BCI Dataset')
    parser.add_argument('--train_dir',   type=str, help='dir path of training data')
    parser.add_argument('--val_dir',     type=str, help='dir path of validation data')
    parser.add_argument('--exp_root',    type=str, help='root dir of experiment')
    parser.add_argument('--config_file', type=str, help='yaml path of configs')
    parser.add_argument('--resume_ckpt', type=str, help='checkpoint path for resuming')
    args = parser.parse_args()

    check_args(args)
    main(args)
