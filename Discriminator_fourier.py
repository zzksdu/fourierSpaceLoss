import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLoss(nn.Module):

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):

        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):

        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight
class FC(nn.Module):
    def __init__(self, inc, outc, activation=nn.ReLU, batch_norm=False):
        super(FC, self).__init__()
        self.fc = nn.Linear(int(inc), int(outc), bias=(not batch_norm))
        self.activation = activation() if activation else None
        self.bn = nn.BatchNorm1d(outc) if batch_norm else None

    def forward(self, x):
        x = self.fc(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class Discriminator_fourier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_list = nn.ModuleList()
        self.fc_list.append(FC(256 * 256 * 6, 1024, batch_norm=True))
        self.fc_list.append(FC(1024, 1024, batch_norm=True))
        self.fc_list.append(FC(1024, 1024, batch_norm=True))
        self.fc_list.append(FC(1024, 1024, batch_norm=True))
        self.fc_list.append(FC(1024, 1, batch_norm=False))

    def forward(self, x):
        bs, _, _, _ = x.size()
        x = x.view(bs, -1)
        for layer in self.fc_list:
            x = layer(x)
        return x



if __name__ == "__main__":

    sr = torch.rand([1, 3, 256, 256])
    hr = torch.rand([1, 3, 256, 256])
    D = Discriminator_fourier()
    cri_gan = GANLoss()
    fake_d_pred = D(sr).detach()
    real_d_pred = D(hr)
    l_d_real = cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
    l_d_real.backward()

    # fake_d_pred = self.net_d(self.output.detach())
    fake_d_pred = D(sr.detach())
    l_d_fake = cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
    l_d_fake.backward()
