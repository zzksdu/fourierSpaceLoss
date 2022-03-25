import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

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

class fourier_Dloss(nn.Module):
    def __init__(self):
        super(fourier_Dloss, self).__init__()
        self.net_df = Discriminator_fourier()
        self.cri_gan = GANLoss()

    def tensor2fft(self, image):
        frm_list = []
        fre_quen = []
        for i in range(3):
            in_ = image[:, i:i + 1, :, :]
            fftn = fft.fftn(in_, dim=(2, 3))
            # add shift
            fftn_shift = fft.fftshift(fftn) #+ 1e-8
            #print('fftn:', fftn_shift.size())
            fre_m = torch.abs(fftn_shift)[:,:, :, :]
            fre_p = torch.angle(fftn_shift)[:,:, :, :]
            frm_list.append(fre_m)
            fre_quen.append(fre_p)
        fre_mag = torch.cat(frm_list, dim=1)
        fre_ang = torch.cat(fre_quen, dim=1)
        return torch.cat([fre_mag, fre_ang], dim=1) # 1 * 256 * 256 * 6

    def comFouier(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h).reshape(h, 1).cuda()
        win2 = torch.hann_window(w).reshape(1, w).cuda()
        win = torch.mm(win1, win2)
        sr_w, hr_w = sr * win, hr * win
        sr_fre = self.tensor2fft(sr_w)
        hr_fre = self.tensor2fft(hr_w)
        return sr_fre, hr_fre

    def forward(self, gt, output):
        gt_fourier = self.comFouier(gt)
        sr_fourier = self.comFouier(output)
        real_d_pred = self.net_df(gt_fourier).detach()
        fake_g_pred = self.net_df(sr_fourier)
        l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
        l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
        l_g_gan_df = (l_g_real + l_g_fake) / 2
        return l_g_gan_df



if __name__ == "__main__":

    sr = torch.rand([1, 3, 256, 256])
    hr = torch.rand([1, 3, 256, 256])

    fourier_Dloss = fourier_Dloss().forward(hr, sr)
    print(fourier_Dloss)
