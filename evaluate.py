import os
import argparse

from libs.utils import *
from libs.evaluate import *
from omegaconf import OmegaConf


def main(args):

    # loads configs
    apply_tta = args.apply_tta
    configs = OmegaConf.load(args.config_file)
    output_dir = os.path.join(args.output_root, configs.exp, args.model_name)
    model_path = os.path.join(args.exp_root, configs.exp, f'{args.model_name}.pth')
    if not os.path.isfile(model_path):
        print(f'{model_path} is not exist', '\n')

    # prints information
    print('-' * 88)
    print('Evaluation for BCI Dataset ...\n')
    print(f'- Data Dir  : {args.data_dir}')
    print(f'- Model Path: {model_path}')
    print(f'- Configs   : {args.config_file}')
    print(f'- Apply TTA : {args.apply_tta}', '\n')

    # initializes evaluator
    if args.evaluator == 'basic':
        evaluator = BCIEvaluatorBasic(configs, model_path, apply_tta)
    elif args.evaluator == 'cahr':
        evaluator = BCIEvaluatorCAHR(configs, model_path, apply_tta)

    # generates predictions
    evaluator.forward(args.data_dir, output_dir)

    print('-' * 88, '\n')
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation for BCI Dataset')
    parser.add_argument('--data_dir',    type=str, help='dir path of data')
    parser.add_argument('--exp_root',    type=str, help='root dir of experiment')
    parser.add_argument('--output_root', type=str, help='root dir of outputs')
    parser.add_argument('--config_file', type=str, help='yaml path of configs')
    parser.add_argument('--model_name',  type=str, help='name of model')
    parser.add_argument('--evaluator',   type=str, help='evaluator type, basic or cahr', default='basic')
    parser.add_argument('--apply_tta',   type=lambda x: (str(x).lower() == 'true'),
                        help='if apply test-time augmentation', default=False)

    args = parser.parse_args()

    check_evaluate_args(args)
    main(args)
