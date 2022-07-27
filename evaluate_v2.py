import os
import argparse

from libs.utils import *
from libs.evaluate_v2 import *
from omegaconf import OmegaConf


def main(args):

    # loads configs
    apply_tta = args.apply_tta
    configs = OmegaConf.load(args.config_file)
    output_dir = os.path.join(args.output_root, configs.exp, args.model_name)
    model_path = os.path.join(args.exp_root, configs.exp, f'{args.model_name}.pth')

    # prints information
    print('-' * 100)
    print('Evaluation for BCI Dataset ...\n')
    print(f'- Data Dir:   {args.data_dir}')
    print(f'- Model Path: {model_path}')
    print(f'- Configs:    {args.config_file}')
    print(f'- Apply TTA:  {args.apply_tta}', '\n')

    # initializes trainer
    evaluator = BCIEvaluator(configs, model_path, apply_tta)

    # generates predictions
    evaluator.forward(args.data_dir, output_dir)

    print('-' * 100, '\n')
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation for BCI Dataset')
    parser.add_argument('--data_dir',    type=str, help='dir path of data')
    parser.add_argument('--exp_root',    type=str, help='root dir of experiment')
    parser.add_argument('--output_root', type=str, help='root dir of outputs')
    parser.add_argument('--config_file', type=str, help='yaml path of configs')
    parser.add_argument('--model_name',  type=str, help='name of model')
    parser.add_argument('--apply_tta',   action='store_true', help='if apply tta')
    args = parser.parse_args()

    check_evaluate_args(args)
    main(args)
