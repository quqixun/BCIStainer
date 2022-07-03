import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

from .logger import *
from .base import BCIBaseTrainer
from torch.cuda.amp import autocast


class BCITrainer(BCIBaseTrainer):

    def __init__(self, configs, exp_dir, resume_ckpt):
        super(BCITrainer, self).__init__(configs, exp_dir, resume_ckpt)

        self.model_name  = configs.model.name
        self.model_prms  = configs.model.params
        self.device      = 'cuda' if torch.cuda.is_available() else 'cpu'

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

    def train_and_val(self, train_loader, val_loader):

        best_val_eval = np.inf
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics   = self._val_epoch(val_loader, epoch)

            # save checkpoint regularly
            if (epoch % self.ckpt_freq == 0) or (epoch + 1 == self.epochs):
                self._save_checkpoint(epoch)

            # save checkpoint with best val loss
            if val_metrics['loss'] < best_val_eval:
                best_val_eval = val_metrics['loss']
                self._save_model()
                print('>>> Best Val Epoch - Lowest Loss - Save Model <<<')

            # write logs
            self._save_logs(epoch, train_metrics, val_metrics)

            print()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        return

    def _train_epoch(self, loader, epoch):
        self.model.train()

        header = 'Train Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)
        logger.add_meter('lr', SmoothedValue(1, '{value:.6f}'))

        data_iter = logger.log_every(loader)
        for iter_step, data in enumerate(data_iter):
            self.optimizer.zero_grad()

            # lr scheduler on per iteration
            if iter_step % self.accum_iter == 0:
                self._adjust_learning_rate(iter_step / len(loader) + epoch)

            he, ihc, level = [d.to(self.device) for d in data]

            with autocast():
                pred_ihc = self.model(he)
                loss = self.mse_loss(ihc, pred_ihc)

            logger.update(
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr']
            )

            loss /= self.accum_iter
            self.scaler.scale(loss).backward()
            if (iter_step + 1) % self.accum_iter == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()

        return {k: meter.global_avg for k, meter in logger.meters.items()}

    @torch.no_grad()
    def _val_epoch(self, loader, epoch):
        self.model.eval()

        header = ' Val  Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)

        data_iter = logger.log_every(loader)
        for _, data in enumerate(data_iter):
            he, ihc, level = [d.to(self.device) for d in data]

            pred_ihc = self.model(he)
            loss = self.mse_loss(ihc, pred_ihc)
            logger.update(loss=loss)

        return {k: meter.global_avg for k, meter in logger.meters.items()}
