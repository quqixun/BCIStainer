# https://github.com/marcelsan/Deep-HdrReconstruction

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class VGG16FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])
        self.layers = 3

        for i in range(self.layers):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(self.layers):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            results.append(func(results[-1]))
        return results[1:]


class VGG19FeatureExtractor(nn.Module):

    def __init__(self, layers=5):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)

        self.enc_1 = torch.nn.Sequential(*vgg19.features[:2])
        self.enc_2 = torch.nn.Sequential(*vgg19.features[2:7])
        self.enc_3 = torch.nn.Sequential(*vgg19.features[7:12])
        self.enc_4 = torch.nn.Sequential(*vgg19.features[12:21])
        self.enc_5 = torch.nn.Sequential(*vgg19.features[21:30])
        self.layers = layers

        for i in range(self.layers):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(self.layers):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            results.append(func(results[-1]))
        return results[1:]


def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def normalize_batch(batch):
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std


class PerceptualLossBase(nn.Module):

    def __init__(self, extractor):
        super().__init__()
        self.extractor = extractor

    def forward(self, fake, real, face, mouth):
        raise NotImplementedError()


class HDRLoss1(PerceptualLossBase):

    def __init__(self, extractor):
        PerceptualLossBase.__init__(self, extractor)
        self.l1 = nn.L1Loss()

    def forward(self, fake, real, face, mouth=None):
        recon_loss = self.l1(fake, real)
        recon_loss += self.l1(fake * face, real * face)

        # Extract features maps.
        feat_fake = self.extractor(normalize_batch(fake))
        feat_real = self.extractor(normalize_batch(real))

        # Calculate VGG loss.
        prc_loss = 0.0
        for i in range(self.extractor.layers):
            prc_loss += self.l1(feat_fake[i], feat_real[i])

        # Calculate style loss.
        style_loss = 0.0
        for i in range(self.extractor.layers):
            style_loss += self.l1(gram_matrix(feat_fake[i]),
                                  gram_matrix(feat_real[i]))

        loss = 6.0 * recon_loss + 1.0 * prc_loss + 120.0 * style_loss

        return loss


class HDRLoss2(PerceptualLossBase):

    def __init__(self, extractor):
        PerceptualLossBase.__init__(self, extractor)
        self.l1 = nn.L1Loss(reduction='none')
        self.l2 = nn.MSELoss(reduction='none')
        self.weights = [1.0, 0.8, 0.6, 0.4, 0.2]
        # self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, fake, real, face, mouth):

        recon_loss = self.l1(fake, real).mean()
        recon_loss += self.l1(fake * face, real * face).sum() / face.sum()
        recon_loss += self.l1(fake * mouth, real * mouth).sum() / mouth.sum()

        # Extract features maps.
        feat_fake = self.extractor(normalize_batch(fake))
        feat_real = self.extractor(normalize_batch(real))

        # Calculate VGG loss and style loss.
        prc_loss, style_loss = 0.0, 0.0
        for i in range(self.extractor.layers):

            rsize = feat_fake[i].size(-1)
            face_tmp = F.interpolate(face, size=rsize, mode='nearest')
            mouth_tmp = F.interpolate(mouth, size=rsize, mode='nearest')

            prc_loss += self.l1(
                feat_fake[i] * face_tmp,
                feat_real[i] * face_tmp
            ).sum() / face_tmp.sum() * self.weights[i]
            prc_loss = prc_loss.sum() / face_tmp.sum()  # wrong
            prc_loss += self.l1(
                feat_fake[i] * mouth_tmp,
                feat_real[i] * mouth_tmp
            ).sum() / mouth_tmp.sum() * self.weights[i]

            style_loss += self.l1(
                gram_matrix(feat_fake[i] * face_tmp),
                gram_matrix(feat_real[i] * face_tmp)
            ).mean()
            style_loss += self.l1(
                gram_matrix(feat_fake[i] * mouth_tmp),
                gram_matrix(feat_real[i] * mouth_tmp)
            ).mean()

        loss = 5.0 * recon_loss + 0.1 * prc_loss + 10.0 * style_loss

        return loss


class HDRLoss3(PerceptualLossBase):

    def __init__(self, extractor):
        PerceptualLossBase.__init__(self, extractor)
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, fake, real, face, mouth):

        recon_loss = self.l1(fake, real)
        recon_loss += self.l1(fake * face, real * face)
        recon_loss += self.l1(fake * mouth, real * mouth)

        # Extract features maps.
        feat_fake = self.extractor(normalize_batch(fake))
        feat_real = self.extractor(normalize_batch(real))

        # Calculate VGG loss and style loss.
        prc_loss, style_loss = 0.0, 0.0
        for i in range(self.extractor.layers):

            rsize = feat_fake[i].size(-1)
            face_tmp = F.interpolate(face, size=rsize, mode='nearest')
            mouth_tmp = F.interpolate(mouth, size=rsize, mode='nearest')

            feat_fake_face = feat_fake[i] * face_tmp
            feat_real_face = feat_real[i] * face_tmp
            feat_fake_mouth = feat_fake[i] * mouth_tmp
            feat_real_mouth = feat_real[i] * mouth_tmp

            prc_loss += self.l1(feat_fake[i], feat_real[i])
            prc_loss += self.l1(feat_fake_face, feat_real_face)
            prc_loss += self.l1(feat_fake_mouth, feat_real_mouth)

            style_loss += self.l1(gram_matrix(feat_fake[i]), gram_matrix(feat_real[i]))
            style_loss += self.l1(gram_matrix(feat_fake_face), gram_matrix(feat_real_face))
            style_loss += self.l1(gram_matrix(feat_fake_mouth), gram_matrix(feat_real_mouth))

        loss = 6.0 * recon_loss + 1.0 * prc_loss + 120.0 * style_loss

        return loss


class HDRLoss4(PerceptualLossBase):

    def __init__(self, extractor):
        PerceptualLossBase.__init__(self, extractor)

        self.l1 = nn.L1Loss(reduction='none')
        self.l2 = nn.MSELoss(reduction='none')

        assert self.extractor.layers == 3
        self.layer_weights = [1.0, 0.1, 0.01]

        return

    def forward(self, fake, real, face, mouth):

        recon_loss = self.l1(fake, real).mean()
        recon_loss += self.l1(fake * face, real * face).sum() / face.sum()
        recon_loss += self.l1(fake * mouth, real * mouth).sum() / mouth.sum()

        # Extract features maps.
        feat_fake = self.extractor(normalize_batch(fake))
        feat_real = self.extractor(normalize_batch(real))

        # Calculate VGG loss and style loss.
        prc_loss, style_loss = 0.0, 0.0
        for i in range(self.extractor.layers):

            rsize = feat_fake[i].size(-1)
            face_tmp = F.interpolate(face, size=rsize, mode='nearest')
            mouth_tmp = F.interpolate(mouth, size=rsize, mode='nearest')

            feat_fake_face = feat_fake[i] * face_tmp
            feat_real_face = feat_real[i] * face_tmp
            feat_fake_mouth = feat_fake[i] * mouth_tmp
            feat_real_mouth = feat_real[i] * mouth_tmp

            prc_loss += self.l1(feat_fake[i], feat_real[i]).mean()
            prc_loss += self.l1(feat_fake_face, feat_real_face).sum() / face_tmp.sum()
            prc_loss += self.l1(feat_fake_mouth, feat_real_mouth).sum() / mouth_tmp.sum()
            prc_loss *= self.layer_weights[i]

            style_loss += self.l1(gram_matrix(feat_fake[i]), gram_matrix(feat_real[i])).mean()
            style_loss += self.l1(gram_matrix(feat_fake_face), gram_matrix(feat_real_face)).mean()
            style_loss += self.l1(gram_matrix(feat_fake_mouth), gram_matrix(feat_real_mouth)).mean()

        # print(recon_loss.item(), prc_loss.item(), style_loss.item())
        loss = 6.0 * recon_loss + 1.0 * prc_loss + 120.0 * style_loss

        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        assert reduction in ['none', 'mean', 'sum']

        self.eps = eps
        self.reduction = reduction

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt((diff * diff) + (self.eps * self.eps))
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss


class HDRLoss5(PerceptualLossBase):

    def __init__(self, extractor):
        PerceptualLossBase.__init__(self, extractor)
        self.charb_l1 = CharbonnierLoss(eps=1e-3, reduction='mean')

    def forward(self, fake, real, face, mouth):

        recon_loss = self.charb_l1(fake, real)
        recon_loss += self.charb_l1(fake * face, real * face)
        recon_loss += self.charb_l1(fake * mouth, real * mouth)

        # Extract features maps.
        feat_fake = self.extractor(normalize_batch(fake))
        feat_real = self.extractor(normalize_batch(real))

        # Calculate VGG loss and style loss.
        prc_loss, style_loss = 0.0, 0.0
        for i in range(self.extractor.layers):

            rsize = feat_fake[i].size(-1)
            face_tmp = F.interpolate(face, size=rsize, mode='nearest')
            mouth_tmp = F.interpolate(mouth, size=rsize, mode='nearest')

            feat_fake_face = feat_fake[i] * face_tmp
            feat_real_face = feat_real[i] * face_tmp
            feat_fake_mouth = feat_fake[i] * mouth_tmp
            feat_real_mouth = feat_real[i] * mouth_tmp

            prc_loss += self.charb_l1(feat_fake[i], feat_real[i])
            prc_loss += self.charb_l1(feat_fake_face, feat_real_face)
            prc_loss += self.charb_l1(feat_fake_mouth, feat_real_mouth)

            style_loss += self.charb_l1(gram_matrix(feat_fake[i]), gram_matrix(feat_real[i]))
            style_loss += self.charb_l1(gram_matrix(feat_fake_face), gram_matrix(feat_real_face))
            style_loss += self.charb_l1(gram_matrix(feat_fake_mouth), gram_matrix(feat_real_mouth))

        loss = 6.0 * recon_loss + 1.0 * prc_loss + 120.0 * style_loss

        return loss


class HDRLoss6(PerceptualLossBase):

    def __init__(self, extractor):
        PerceptualLossBase.__init__(self, extractor)

        self.charb_l1 = CharbonnierLoss(eps=1e-3, reduction='none')
        assert self.extractor.layers == 3
        self.layer_weights = [1.0, 0.1, 0.01]

        return

    def forward(self, fake, real, face, mouth):

        recon_loss = self.charb_l1(fake, real).mean()
        recon_loss += self.charb_l1(fake * face, real * face).sum() / face.sum()
        recon_loss += self.charb_l1(fake * mouth, real * mouth).sum() / mouth.sum()

        # Extract features maps.
        feat_fake = self.extractor(normalize_batch(fake))
        feat_real = self.extractor(normalize_batch(real))

        # Calculate VGG loss and style loss.
        prc_loss, style_loss = 0.0, 0.0
        for i in range(self.extractor.layers):

            rsize = feat_fake[i].size(-1)
            face_tmp = F.interpolate(face, size=rsize, mode='nearest')
            mouth_tmp = F.interpolate(mouth, size=rsize, mode='nearest')

            feat_fake_face = feat_fake[i] * face_tmp
            feat_real_face = feat_real[i] * face_tmp
            feat_fake_mouth = feat_fake[i] * mouth_tmp
            feat_real_mouth = feat_real[i] * mouth_tmp

            prc_loss += self.charb_l1(feat_fake[i], feat_real[i]).mean()
            prc_loss += self.charb_l1(feat_fake_face, feat_real_face).sum() / face_tmp.sum()
            prc_loss += self.charb_l1(feat_fake_mouth, feat_real_mouth).sum() / mouth_tmp.sum()
            prc_loss *= self.layer_weights[i]

            style_loss += self.charb_l1(gram_matrix(feat_fake[i]), gram_matrix(feat_real[i])).mean()
            style_loss += self.charb_l1(gram_matrix(feat_fake_face), gram_matrix(feat_real_face)).mean()
            style_loss += self.charb_l1(gram_matrix(feat_fake_mouth), gram_matrix(feat_real_mouth)).mean()

        loss = 6.0 * recon_loss + 1.0 * prc_loss + 120.0 * style_loss

        return loss


class HDRLoss7(PerceptualLossBase):

    def __init__(self, extractor):
        PerceptualLossBase.__init__(self, extractor)
        self.charb_l1 = CharbonnierLoss(eps=1e-3, reduction='mean')

    def forward(self, fake, real, face, mouth):

        recon_loss = self.charb_l1(fake, real)
        recon_loss += self.charb_l1(fake * face, real * face)
        recon_loss += self.charb_l1(fake * mouth, real * mouth)

        # Extract features maps.
        feat_fake = self.extractor(normalize_batch(fake))
        feat_real = self.extractor(normalize_batch(real))

        # Calculate VGG loss.
        prc_loss = 0.0
        for i in range(self.extractor.layers):

            rsize = feat_fake[i].size(-1)
            face_tmp = F.interpolate(face, size=rsize, mode='nearest')
            mouth_tmp = F.interpolate(mouth, size=rsize, mode='nearest')

            feat_fake_face = feat_fake[i] * face_tmp
            feat_real_face = feat_real[i] * face_tmp
            feat_fake_mouth = feat_fake[i] * mouth_tmp
            feat_real_mouth = feat_real[i] * mouth_tmp

            prc_loss += self.charb_l1(feat_fake[i], feat_real[i])
            prc_loss += self.charb_l1(feat_fake_face, feat_real_face)
            prc_loss += self.charb_l1(feat_fake_mouth, feat_real_mouth)

        loss = 6.0 * recon_loss + 1.0 * prc_loss

        return loss


def load_hdrloss(mode, extractor_name='vgg16', vgg19_layers=5):

    # assert mode in ['hdr1', 'hdr2', 'hdr3', 'hdr4', 'hdr5', 'hdr6']
    assert extractor_name in ['vgg16', 'vgg19']

    if extractor_name == 'vgg16':
        extractor = VGG16FeatureExtractor()
    else:  # extractor_name == 'vgg19
        extractor = VGG19FeatureExtractor(vgg19_layers)

    if mode == 'hdr1':
        return HDRLoss1(extractor)
    elif mode == 'hdr2':
        return HDRLoss2(extractor)
    elif mode == 'hdr3':
        return HDRLoss3(extractor)
    elif mode == 'hdr4':
        return HDRLoss4(extractor)
    elif mode == 'hdr5':
        return HDRLoss5(extractor)
    elif mode == 'hdr6':
        return HDRLoss6(extractor)
    elif mode == 'hdr7':
        return HDRLoss7(extractor)
