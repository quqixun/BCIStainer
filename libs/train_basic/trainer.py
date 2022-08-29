import time
import torch
import datetime
import torch.nn.functional as F

from ..utils import *


class BCIBasicTrainer(BCIBaseTrainer):

    def __init__(self, configs, exp_dir, resume_ckpt):
        super(BCIBasicTrainer, self).__init__(configs, exp_dir, resume_ckpt)

    def forward(self, train_loader, val_loader):

        best_val_psnr = 0.0
        best_val_ssim = 0.0
        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_epoch(train_loader, epoch)

            # save model with best val psnr
            val_model = self.Gema if self.ema else self.G
            val_metrics = self._val_epoch(val_model, val_loader, epoch)
            if val_metrics['psnr'] > best_val_psnr:
                best_val_psnr = val_metrics['psnr']
                self._save_model(val_model, 'best_psnr')
                print('>>> Best Val Epoch - Highest PSNR - Save Model <<<')
                best_psnr_msg = f'- Best PSNR:{best_val_psnr:.4f} in Epoch:{epoch}'

            if val_metrics['ssim'] > best_val_ssim:
                best_val_ssim = val_metrics['ssim']
                self._save_model(val_model, 'best_ssim')
                print('>>> Best Val Epoch - Highest SSIM - Save Model <<<')
                best_ssim_msg = f'- Best SSIM:{best_val_ssim:.4f} in Epoch:{epoch}'

            # save checkpoint regularly
            if (epoch % self.ckpt_freq == 0) or (epoch + 1 == self.epochs):
                self._save_checkpoint(epoch)

            # write logs
            self._save_logs(epoch, train_metrics, val_metrics)
            print()

        print(best_psnr_msg)
        print(best_ssim_msg)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('- Training time {}'.format(total_time_str))

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

            # forward
            he, ihc, level = [d.to(self.device) for d in data]
            multi_outputs = self.G(he)
            if not self.G.output_lowres:
                ihc_hr_pred, level_pred = multi_outputs
                ihc_lr_pred = None
            else:  # self.G.output_lowres is True
                ihc_hr_pred, ihc_lr_pred, level_pred = multi_outputs

            # update D
            self._set_requires_grad(self.D, True)
            D_fake, D_real = self._D_loss(he, ihc, ihc_hr_pred)
            loss_D = (D_fake + D_real) * 0.5
            loss_D.backward()
            if (iter_step + 1) % self.accum_iter == 0:
                self.D_opt.step()

            # update G
            self._set_requires_grad(self.D, False)
            G_gan, G_rec, G_sim = self._G_loss(he, ihc, ihc_hr_pred, ihc_lr_pred)
            G_cls = self.cls_loss(level, level_pred)
            loss_G = G_gan + G_rec + G_sim + G_cls
            loss_G.backward()
            if (iter_step + 1) % self.accum_iter == 0:
                self.G_opt.step()

            # update logger
            logger.update(
                D_fake=D_fake.item(),
                D_real=D_real.item(),
                G_gan=G_gan.item(),
                G_rec=G_rec.item(),
                G_sim=G_sim.item(),
                G_cls=G_cls.item(),
                lr=self.G_opt.param_groups[0]['lr']
            )

            if self.ema:
                self.Gema.update()

        logger_info = {
            key: meter.global_avg
            for key, meter in logger.meters.items()
        }
        return logger_info

    @torch.no_grad()
    def _val_epoch(self, val_model, loader, epoch):

        val_model.eval()

        header = ' Val  Epoch:[{}]'.format(epoch)
        logger = MetricLogger(header, self.print_freq)

        data_iter = logger.log_every(loader)
        for _, data in enumerate(data_iter):
            he, ihc, _ = [d.to(self.device) for d in data]
            multi_outputs = val_model(he)
            ihc_hr_pred = multi_outputs[0]

            psnr, ssim = self.eval_metrics(ihc, ihc_hr_pred)
            logger.update(psnr=psnr.item(), ssim=ssim.item())
        
        logger_info = {
            key: meter.global_avg
            for key, meter in logger.meters.items()
        }
        return logger_info

    def _D_loss(self, he, ihc, ihc_hr_pred):

        # fake
        fake = self._get_D_input(he, ihc_hr_pred)
        pred_fake = self.D(fake.detach())
        D_fake = self.gan_loss(pred_fake, False, for_D=True)

        # real
        real = self._get_D_input(he, ihc)
        pred_real = self.D(real)
        D_real = self.gan_loss(pred_real, True, for_D=True)

        return D_fake, D_real

    def _G_loss(self, he, ihc, ihc_hr_pred, ihc_lr_pred=None):

        # gan
        fake = self._get_D_input(he, ihc_hr_pred)
        pred_fake = self.D(fake)
        G_gan = self.gan_loss(pred_fake, True, for_D=False)

        # rec
        G_rec = self.rec_loss(ihc, ihc_hr_pred)
        if (ihc_lr_pred is not None) and (self.low_weight > 0):
            _, _, h, w = ihc_lr_pred.size()
            ihc_low = F.interpolate(ihc, size=(h, w), mode='bilinear', align_corners=True)
            G_rec_low = self.rec_loss(ihc_low, ihc_lr_pred)
            G_rec += G_rec_low * self.low_weight

        # sim
        G_sim = self.sim_loss(ihc, ihc_hr_pred)

        return G_gan, G_rec, G_sim

    def _get_D_input(self, he, ihc):

        if self.D_input == 'he+ihc':
            D_input = torch.cat((he, ihc), 1)
        else:  # self.D_input == 'ihc
            D_input = ihc
        
        if self.diffaug:
            D_input = DiffAugment(D_input)

        return D_input
