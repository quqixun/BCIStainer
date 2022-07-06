import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

from .logger import *
from .base import BCIBaseTrainer


class BCITrainer(BCIBaseTrainer):

    def __init__(self, configs, exp_dir, resume_ckpt):
        super(BCITrainer, self).__init__(configs, exp_dir, resume_ckpt)

    def forward(self, train_loader, val_loader):

        best_val_eval = np.inf
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_epoch(train_loader, epoch)
            val_metrics   = self._val_epoch(val_loader, epoch)

            # save checkpoint regularly
            if (epoch % self.ckpt_freq == 0) or (epoch + 1 == self.epochs):
                self._save_checkpoint(epoch)

            # save model with best val loss
            if val_metrics['total'] < best_val_eval:
                best_val_eval = val_metrics['total']
                self._save_model('best')
                print('>>> Best Val Epoch - Lowest Total Loss - Save Model <<<')

            # save latest model
            self._save_model('latest')

            # write logs
            self._save_logs(epoch, train_metrics, val_metrics)

            print()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        return

    def _train_epoch(self, loader, epoch):
        self.D.train()
        self.G.train()

        header = 'Train Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)
        logger.add_meter('lr', SmoothedValue(1, '{value:.6f}'))

        data_iter = logger.log_every(loader)
        for iter_step, data in enumerate(data_iter):
            self.D_opt.zero_grad()
            self.G_opt.zero_grad()

            # lr scheduler on per iteration
            if iter_step % self.accum_iter == 0:
                self._adjust_learning_rate(iter_step / len(loader) + epoch)

            he, ihc, level = [d.to(self.device) for d in data]
            ihc_pred = self.G(he)

            # update D
            self._set_requires_grad(self.D, True)
            D_fake, D_real = self._D_loss(he, ihc, ihc_pred)
            loss_D = (D_fake + D_real) * 0.5
            loss_D /= self.accum_iter
            loss_D.backward()
            if (iter_step + 1) % self.accum_iter == 0:
                self.D_opt.step()

            # update G
            self._set_requires_grad(self.D, False)
            G_gan, G_rec = self._G_loss(he, ihc, ihc_pred)
            loss_G = G_gan + G_rec
            loss_G /= self.accum_iter
            loss_G.backward()
            if (iter_step + 1) % self.accum_iter == 0:
                self.G_opt.step()

            logger.update(
                D_fake=D_fake.item(),
                D_real=D_real.item(),
                G_gan=G_gan.item(),
                G_rec=G_rec.item(),
                lr=self.G_opt.param_groups[0]['lr']
            )

        return {k: meter.global_avg for k, meter in logger.meters.items()}

    @torch.no_grad()
    def _val_epoch(self, loader, epoch):
        self.D.eval()
        self.G.eval()

        header = ' Val  Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)

        data_iter = logger.log_every(loader)
        for _, data in enumerate(data_iter):
            he, ihc, level = [d.to(self.device) for d in data]
            ihc_pred = self.G(he)

            D_fake, D_real = self._D_loss(he, ihc, ihc_pred)
            G_gan, G_rec = self._G_loss(he, ihc, ihc_pred)
            total = (D_fake + D_real) * 0.5 + G_gan + G_rec

            logger.update(
                D_fake=D_fake.item(),
                D_real=D_real.item(),
                G_gan=G_gan.item(),
                G_rec=G_rec.item(),
                total=total.item()
            )

        return {k: meter.global_avg for k, meter in logger.meters.items()}

    def _D_loss(self, he, ihc, ihc_pred):

        # fake
        fake = torch.cat((he, ihc_pred), 1)
        pred_fake = self.D(fake.detach())
        D_fake = self.gan_loss(pred_fake, False)

        # real
        real = torch.cat((he, ihc), 1)
        pred_real = self.D(real)
        D_real = self.gan_loss(pred_real, True)

        return D_fake, D_real

    def _G_loss(self, he, ihc, ihc_pred):

        # gan loss
        fake = torch.cat((he, ihc_pred), 1)
        pred_fake = self.D(fake)
        G_gan = self.gan_loss(pred_fake, True)

        # rec
        G_rec = self.rec_loss(ihc, ihc_pred)

        return G_gan, G_rec
