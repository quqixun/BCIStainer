import time
import torch
import datetime
import torch.nn.functional as F

from ..utils import *


class BCICAHRTrainer(BCIBaseTrainer):

    def __init__(self, configs, exp_dir, resume_ckpt):
        super(BCICAHRTrainer, self).__init__(configs, exp_dir, resume_ckpt)
        self.crop_loss  = self.configs.trainer.crop_loss
        self.infer_mode = self.configs.trainer.infer_mode

    def forward(self, train_loader, val_loader):

        best_val_psnr = 0.0
        best_val_clsf = np.inf
        start_time = time.time()

        basic_msg = 'PSNR:{:.4f} SSIM:{:.4f} CLSF:{:.4f} Epoch:{}'
        for epoch in range(self.start_epoch, self.epochs):
            train_metrics = self._train_epoch(train_loader, epoch)

            # save model with best val psnr
            val_model = self.Gema if self.ema else self.G
            val_metrics = self._val_epoch(val_model, val_loader, epoch)
            psnr = val_metrics['psnr']
            ssim = val_metrics['ssim']
            clsf = val_metrics['clsf'] if self.apply_cmp else 0.0
            info_list = [psnr, ssim, clsf, epoch]

            if psnr > best_val_psnr:
                best_val_psnr = psnr
                self._save_model(val_model, 'best_psnr')
                print('>>> Highest PSNR - Save Model <<<')
                psnr_msg = '- Best PSNR: ' + basic_msg.format(*info_list)

            if self.apply_cmp:
                if clsf < best_val_clsf:
                    best_val_clsf = clsf
                    self._save_model(val_model, 'best_clsf')
                    print('>>> Lowest  CLSF - Save Model <<<')
                    clsf_msg = '- Best CLSF: ' + basic_msg.format(*info_list)

            # save checkpoint regularly
            if (epoch % self.ckpt_freq == 0) or (epoch + 1 == self.epochs):
                self._save_checkpoint(epoch)

            # write logs
            self._save_logs(epoch, train_metrics, val_metrics)
            print()

        print(psnr_msg)
        if self.apply_cmp:
            print(clsf_msg)

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
            logger.update(lr=self.G_opt.param_groups[0]['lr'])

            # forward
            he, ihc, level, he_crop, ihc_crop, crop_idx = [d.to(self.device) for d in data]
            multi_outputs = self.G(he, he_crop, crop_idx, mode='train')
            if not self.G.output_lowres:
                ihc_hr_pred, ihc_crop_pred, level_pred = multi_outputs
                ihc_lr_pred = None
            else:  # self.G.output_lowres is True
                ihc_hr_pred, ihc_lr_pred, ihc_crop_pred, level_pred = multi_outputs

            # update C
            if self.apply_cmp and (epoch < self.start_cmp):
                self.C.train()
                self._set_requires_grad(self.C, True)
                ihc_level, ihc_latent = self.C(ihc)
                loss_C = self.ccl_loss(level, ihc_level)
                logger.update(Cc=loss_C.item())
                loss_C.backward()
                if (iter_step + 1) % self.accum_iter == 0:
                    self.C_opt.step()

            # update D
            self._set_requires_grad(self.D, True)
            D_fake, D_real = self._D_loss(he, ihc, ihc_hr_pred)
            if self.crop_loss:
                D_crop_fake, D_crop_real = self._D_loss(he_crop, ihc_crop, ihc_crop_pred)
                D_fake += D_crop_fake
                D_real += D_crop_real
            logger.update(Df=D_fake.item(), Dr=D_real.item())
            loss_D = (D_fake + D_real) * 0.5
            loss_D.backward()
            if (iter_step + 1) % self.accum_iter == 0:
                self.D_opt.step()

            # update G
            self._set_requires_grad(self.D, False)
            G_gan, G_rec, G_sim = self._G_loss(he, ihc, ihc_hr_pred, ihc_lr_pred)
            if self.crop_loss:
                G_crop_gan, G_crop_rec, G_crop_sim = self._G_loss(he_crop, ihc_crop, ihc_crop_pred)
                G_gan += G_crop_gan
                G_rec += G_crop_rec
                G_sim += G_crop_sim
            G_cls = self.gcl_loss(level, level_pred)
            logger.update(Gg=G_gan.item(), Gr=G_rec.item(),
                          Gs=G_sim.item(), Gc=G_cls.item())
            loss_G = G_gan + G_rec + G_sim + G_cls

            if self.apply_cmp and (epoch >= self.start_cmp):
                self.C.eval()
                self._set_requires_grad(self.C, False)
                ihc_level, ihc_latent = self.C(ihc)
                G_cmp = self._C_loss(ihc_latent, ihc_hr_pred, level)
                logger.update(Gm=G_cmp.item())
                loss_G += G_cmp

            loss_G.backward()
            if (iter_step + 1) % self.accum_iter == 0:
                self.G_opt.step()
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
            he, ihc, level, he_crop, crop_idx = [d.to(self.device) for d in data]
            he_crop, crop_idx = he_crop[0], crop_idx[0]
            multi_outputs = self.G(he, he_crop, crop_idx, self.infer_mode)
            ihc_hr_pred = multi_outputs[0]

            psnr, ssim = self.eval_metrics(ihc, ihc_hr_pred)
            logger.update(psnr=psnr.item(), ssim=ssim.item())

            if self.apply_cmp:
                self.C.eval()
                ihc_level, ihc_latent = self.C(ihc_hr_pred)
                clsf = self.ccl_loss(level, ihc_level)
                logger.update(clsf=clsf.item())

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

    def _C_loss(self, ihc_latent, ihc_hr_pred, level):

        ihc_pred_level, ihc_pred_latent = self.C(ihc_hr_pred)
        if self.cmp_loss.mode == 'csim':
            G_cmp = self.cmp_loss(ihc_latent.detach(), ihc_pred_latent)
        else:  # self.cmp_loss.mode in ['ce', 'focal']
            G_cmp = self.cmp_loss(level, ihc_pred_level)

        return G_cmp

    def _get_D_input(self, he, ihc):

        if self.D_input == 'he+ihc':
            D_input = torch.cat((he, ihc), 1)
        else:  # self.D_input == 'ihc
            D_input = ihc
        
        if self.diffaug:
            D_input = DiffAugment(D_input)

        return D_input
