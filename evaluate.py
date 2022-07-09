import os
import argparse

from libs.utils import *
from libs.evaluate import *
from omegaconf import OmegaConf


def main(args):

    print(args.config_file, args.model_name)

    # loads configs
    configs = OmegaConf.load(args.config_file)

    # initializes trainer
    model_file = f'{args.model_name}.pth'
    model_path = os.path.join(args.exp_root, configs.exp, model_file)
    evaluator = BCIEvaluator(configs, model_path)

    # generates predictions
    output_dir = os.path.join(args.output_root, configs.exp, args.model_name)
    evaluator.forward(args.data_dir, output_dir)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation for BCI Dataset')
    parser.add_argument('--data_dir',    type=str, help='dir path of data')
    parser.add_argument('--exp_root',    type=str, help='root dir of experiment')
    parser.add_argument('--output_root', type=str, help='root dir of outputs')
    parser.add_argument('--config_file', type=str, help='yaml path of configs')
    parser.add_argument('--model_name',  type=str, help='name of model')
    args = parser.parse_args()

    check_evaluate_args(args)
    main(args)
