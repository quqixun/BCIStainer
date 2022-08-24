import os
import torch
import random
import numpy as np


HE_STAT = {
    'channel_mean': np.array([[[162.5, 137.9, 159.45]]]),
    'channel_std':  np.array([[[35.8,  43.0,  31.6]]]),
    'global_mean':  153.3,
    'global_std':   38.7,
    'global_min':   0.0,
    'global_max':   255.0
}

IHC_STAT = {
    'channel_mean': np.array([[[196.0, 190.2, 183.6]]]),
    'channel_std':  np.array([[[39.1,  40.7,  44.3]]]),
    'global_mean':  189.9,
    'global_std':   41.7,
    'global_min':   0.0,
    'global_max':   255.0
}


def check_train_args(args):

    if not os.path.isdir(args.train_dir):
        raise IOError(f'train_dir {args.train_dir} is not exist')
    
    if not os.path.isdir(args.val_dir):
        raise IOError(f'val_dir {args.val_dir} is not exist')

    if not os.path.isfile(args.config_file):
        raise IOError(f'config_file {args.config_file} is not exist')

    if args.trainer not in ['basic', 'cahr']:
        raise ValueError('trainer is not one of basic or cahr')

    return


def check_evaluate_args(args):

    if not os.path.isdir(args.data_dir):
        raise IOError(f'data_dir {args.data_dir} is not exist')
    
    if not os.path.isdir(args.exp_root):
        raise IOError(f'exp_root {args.exp_root} is not exist')

    if not os.path.isfile(args.config_file):
        raise IOError(f'config_file {args.config_file} is not exist')

    model_name_list = ['model_best_psnr', 'model_best_ssim', 'model_latest']
    if args.model_name not in model_name_list:
        raise ValueError(f'model_name {args.model_name} is not supportted')

    if args.evaluator not in ['basic', 'cahr']:
        raise ValueError('evaluator is not one of basic or cahr')

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


def normalize_image(image, image_type, norm_method):

    if image_type == 'he':
        stat = HE_STAT
    elif image_type == 'ihc':
        stat = IHC_STAT
    else:
        raise ValueError('unknown image_type')

    if norm_method == 'channel_zscore':
        if image.ndims == 3:
            mean_ = stat['channel_mean']
            std_  = stat['channel_std']
        elif image.ndims == 4:
            mean_ = stat['channel_mean'][None, ...]
            std_  = stat['channel_std'][None, ...]
        image_norm = (image - mean_) / std_
    elif norm_method == 'global_zscore':
        image_norm = (image - stat['global_mean']) / stat['global_std']
    elif norm_method == 'global_minmax':
        global_range =  stat['global_max'] - stat['global_min']
        image_norm = (image - stat['global_min']) / global_range
        image_norm = image_norm * 2.0 - 1.0
    else:
        raise ValueError('unknown norm_method')

    return image_norm


def unnormalize_image(image, image_type, norm_method):

    if image_type == 'he':
        stat = HE_STAT
    elif image_type == 'ihc':
        stat = IHC_STAT
    else:
        raise ValueError('unknown image_type')

    if norm_method == 'channel_zscore':
        if image.ndims == 3:
            mean_ = stat['channel_mean']
            std_  = stat['channel_std']
        elif image.ndims == 4:
            mean_ = stat['channel_mean'][None, ...]
            std_  = stat['channel_std'][None, ...]
        image_unnorm = image * std_ + mean_
    elif norm_method == 'global_zscore':
        image_unnorm = image * stat['global_std'] + stat['global_mean']
    elif norm_method == 'global_minmax':
        image_unnorm = (image + 1.0) / 2.0
        global_range = stat['global_max'] - stat['global_min']
        image_unnorm = image_unnorm * global_range + stat['global_min']
    else:
        raise ValueError('unknown norm_method')

    return image_unnorm


def tta(image, no):
    if no == 0:
        return image
    elif no == 1:
        return np.rot90(image, 1)
    elif no == 2:
        return np.rot90(image, 2)
    elif no == 3:
        return np.rot90(image, 3)
    elif no == 4:
        return np.fliplr(image)
    elif no == 5:
        return np.flipud(image)
    elif no == 6:
        return image.transpose(1, 0, 2)
    else:
        raise NotImplemented('unknown no for tta')


def untta(image, no):
    if no == 0:
        return image
    elif no == 1:
        return np.rot90(image, 3)
    elif no == 2:
        return np.rot90(image, 2)
    elif no == 3:
        return np.rot90(image, 1)
    elif no == 4:
        return np.fliplr(image)
    elif no == 5:
        return np.flipud(image)
    elif no == 6:
        return image.transpose(1, 0, 2)
    else:
        raise NotImplemented('unknown no for untta')
