import os
import json
import math
import torch

from .losses import *
from ema_pytorch import EMA
from ..models import define_G, define_D, define_C


class BCIBaseTrainer(object):

    def __init__(self, configs, exp_dir, resume_ckpt):

        self.configs  = configs
        self.exp_dir  = exp_dir
        self.ckpt_dir = os.path.join(self.exp_dir, 'ckpts')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.log_path = os.path.join(self.exp_dir, 'log.txt')
        self.resume_ckpt = resume_ckpt

        # trainer
        self.start_epoch = 0
        self.epochs      = configs.trainer.epochs
        self.ckpt_freq   = configs.trainer.ckpt_freq
        self.print_freq  = configs.trainer.print_freq
        self.accum_iter  = configs.trainer.accum_iter
        self.diffaug     = configs.trainer.diffaug
        self.ema         = configs.trainer.ema
        self.low_weight  = configs.trainer.low_weight
        self.apply_cmp   = configs.trainer.get('apply_cmp', False)
        self.start_cmp   = configs.trainer.get('start_cmp', 0)

        # model
        self.D_params = configs.D
        self.G_params = configs.G
        self.device   = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.D_input  = 'ihc' if configs.D.params.input_channels == 3 else 'he+ihc'
        if self.apply_cmp:
            self.C_params = configs.C

        # loss
        self.cls_params = configs.loss.cls
        self.rec_params = configs.loss.rec
        self.sim_params = configs.loss.sim
        self.gan_params = configs.loss.gan
        if self.apply_cmp:
            self.cmp_params = configs.loss.cmp

        # optimizer
        self.opt_name   = configs.optimizer.name
        self.opt_params = configs.optimizer.params

        # scheduler
        self.min_lr = configs.scheduler.min_lr
        self.warmup = configs.scheduler.warmup

        self._load_model()
        self._load_losses()
        self._load_optimizer()
        self._load_checkpoint()

    def _load_model(self):

        self.D = define_D(self.D_params)
        self.D = self.D.to(self.device)
        self.G = define_G(self.G_params)
        self.G = self.G.to(self.device)

        if self.ema:
            self.Gema = EMA(
                self.G,
                beta=0.99,
                update_after_step=100,
                update_every=1,
                power=1.0
            )

        if self.apply_cmp:
            self.C = define_C(self.C_params)
            self.C = self.C.to(self.device)

        return

    def _load_losses(self):

        self.gcl_loss = ClsLoss(**self.cls_params).to(self.device)
        self.rec_loss = RecLoss(**self.rec_params).to(self.device)
        self.sim_loss = SimLoss(**self.sim_params).to(self.device)
        self.gan_loss = MSGANLoss(**self.gan_params).to(self.device)

        if self.apply_cmp:
            self.cmp_loss = CmpLoss(**self.cmp_params).to(self.device)
            self.ccl_loss = ClsLoss(mode='focal', weight=1.0).to(self.device)

        self.eval_metrics = EvalMetrics().to(self.device)

        return

    def _load_optimizer(self):

        if self.opt_name == 'Adam':
            opt_func = torch.optim.Adam
        elif self.opt_name == 'AdamW':
            opt_func = torch.optim.AdamW
        elif self.opt_name == 'SGD':
            opt_func = torch.optim.SGD
        else:
            raise ValueError('Unknown optimizer')

        self.D_opt = opt_func(self.D.parameters(), **self.opt_params)
        self.G_opt = opt_func(self.G.parameters(), **self.opt_params)
        if self.apply_cmp:
            self.C_opt = opt_func(self.C.parameters(), lr=1e-4, betas=(0.9, 0.99))

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
            self.D.load_state_dict(checkpoint['D'])
            self.G.load_state_dict(checkpoint['G'])
            self.D_opt.load_state_dict(checkpoint['D_opt'])
            self.G_opt.load_state_dict(checkpoint['G_opt'])
            if self.ema:
                self.Gema.load_state_dict(checkpoint['Gema'])
            if self.apply_cmp:
                self.C.load_state_dict(checkpoint['C'])
                self.C_opt.load_state_dict(checkpoint['C_opt'])

        except Exception:
            print('Faild to resume checkpoint')

        return

    def _save_checkpoint(self, epoch):

        ckpt_file = f'ckpt-{epoch:06d}.pth'
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_file)

        ckpt = {
            'epoch': epoch,
            'D':     self.D.state_dict(),
            'G':     self.G.state_dict(),
            'D_opt': self.D_opt.state_dict(),
            'G_opt': self.G_opt.state_dict(),
        }
        if self.ema:
            ckpt['Gema'] = self.Gema.state_dict()
        if self.apply_cmp:
            ckpt['C']     = self.C.state_dict()
            ckpt['C_opt'] = self.C_opt.state_dict()

        torch.save(ckpt, ckpt_path)

        return

    def _save_model(self, model, model_name):

        model_path = os.path.join(self.exp_dir, f'model_{model_name}.pth')
        torch.save(model.state_dict(), model_path)

        return

    def _save_logs(self, epoch, train_metrics, val_metrics):

        log_stats = {
            **{f't{k}': round(v, 6) for k, v in train_metrics.items()},
            **{f'v{k}': round(v, 6) for k, v in val_metrics.items()},
            'epoch': epoch
        }
        with open(self.log_path, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(log_stats) + '\n')

        return

    def _adjust_learning_rate(self, epoch):

        if epoch < self.warmup:
            lr = self.opt_params.lr * epoch / self.warmup
        else:
            after_warmup = self.epochs - self.warmup
            epoch_ratio = (epoch - self.warmup) / after_warmup
            lr = self.min_lr + \
                (self.opt_params.lr - self.min_lr) * 0.5 * \
                (1.0 + math.cos(math.pi * epoch_ratio))

        optimizers = [self.G_opt, self.D_opt]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                if 'lr_scale' in param_group:
                    param_group['lr'] = lr * param_group['lr_scale']
                else:
                    param_group['lr'] = lr

        return

    def _set_requires_grad(self, nets, requires_grad=False):

        if not isinstance(nets, list):
            nets = [nets]

        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
