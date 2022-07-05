import os
import json
import math
import lpips
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from torch.cuda.amp import GradScaler


class BCIBaseTrainer(object):

    def __init__(self, configs, exp_dir, resume_ckpt):

        self.configs  = configs
        self.exp_dir  = exp_dir
        self.ckpt_dir = os.path.join(self.exp_dir, 'ckpts')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.log_path = os.path.join(self.exp_dir, 'log.txt')
        self.resume_ckpt = resume_ckpt

        # model
        self.model_name = configs.model.name
        self.model_prms = configs.model.params
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'

        # optimizer
        self.opt_name = configs.optimizer.name
        self.opt_prms = configs.optimizer.params

        # scheduler
        self.min_lr = configs.scheduler.min_lr
        self.warmup = configs.scheduler.warmup

        # trainer
        self.start_epoch = 0
        self.epochs      = configs.trainer.epochs
        self.ckpt_freq   = configs.trainer.ckpt_freq
        self.print_freq  = configs.trainer.print_freq
        self.accum_iter  = configs.trainer.accum_iter

        self._load_model()
        self._load_losses()
        self._load_optimizer()
        self._load_checkpoint()

    def _load_model(self):

        if self.model_name == 'UNet':
            model_func = smp.Unet
        else:
            raise ValueError('Unknown model')

        self.model = model_func(**self.model_prms)
        self.model = self.model.to(self.device)

        return

    def _load_losses(self):

        self.mse_loss   = nn.MSELoss()
        self.mae_loss   = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)

        return

    def _load_optimizer(self):

        if self.opt_name == 'Adam':
            opt_func = torch.optim.Adam
        elif self.opt_name == 'AdamW':
            opt_func = torch.optim.AdamW
        else:
            raise ValueError('Unknown optimizer')

        self.optimizer = opt_func(self.model.parameters(), **self.opt_prms)
        self.scaler = GradScaler()

        return

    def _load_checkpoint(self):

        if self.resume_ckpt is None:
            return

        ckpt_path = None
        if os.path.isfile(self.resume_ckpt):
            ckpt_path = self.resume_ckpt
        else:  # find checkpoint from models_dir
            ckpt_files = os.listdir(self.ckpt_dir)
            ckpt_files = [f for f in ckpt_files if f.startswith('ckpt')]
            if len(ckpt_files) > 0:
                ckpt_files.sort()
                ckpt_file = ckpt_files[-1]
                ckpt_path = os.path.join(self.ckpt_dir, ckpt_file)

        if ckpt_path is None:
            return

        try:
            print('Resume checkpoint from:', ckpt_path)
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.start_epoch = checkpoint['epoch'] + 1
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print('Faild to resume checkpoint')

        return

    def _save_checkpoint(self, epoch):

        ckpt_file = f'ckpt-{epoch:06d}.pth'
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file)

        ckpt = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(ckpt, ckpt_path)

        return

    def _save_model(self):

        model_path = os.path.join(self.exp_dir, 'model.pth')
        torch.save(self.model.state_dict(), model_path)

        return

    def _save_logs(self, epoch, train_metrics, val_metrics):

        log_stats = {
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
            'epoch': epoch
        }
        with open(self.log_path, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(log_stats) + '\n')

        return

    def _adjust_learning_rate(self, epoch):

        if epoch < self.warmup:
            lr = self.opt_prms.lr * epoch / self.warmup 
        else:
            after_warmup = self.epochs - self.warmup
            epoch_ratio = (epoch - self.warmup) / after_warmup
            lr = self.min_lr + \
                (self.opt_prms.lr - self.min_lr) * 0.5 * \
                (1.0 + math.cos(math.pi * epoch_ratio))

        for param_group in self.optimizer.param_groups:
            if 'lr_scale' in param_group:
                param_group['lr'] = lr * param_group['lr_scale']
            else:
                param_group['lr'] = lr

        return
