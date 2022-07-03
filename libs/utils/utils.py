import os
import torch
import random
import numpy as np


HE_STAT = {
    'mean': np.array([[[162.46435801, 137.90028199, 159.48469157]]]),
    'std':  np.array([[[35.83228623,  42.97987523,  31.60071293]]])
}

IHC_STAT = {
    'mean': np.array([[[196.00741482, 190.21520278, 183.56929584]]]),
    'std':  np.array([[[39.06072075,  40.73281569,  44.30617855]]])
}


def check_args(args):

    if not os.path.isdir(args.train_dir):
        raise IOError(f'train_dir {args.train_dir} is not exist')
    
    if not os.path.isdir(args.val_dir):
        raise IOError(f'val_dir {args.val_dir} is not exist')

    if not os.path.isfile(args.config_file):
        raise IOError(f'config_file {args.config_file} is not exist')

    return


def init_environment(seed):

    # sets seed for completely reproducible results
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return
