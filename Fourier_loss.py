import torch
import torch.fft as fft
import torch.nn as nn
from torch.nn import functional as F


class FourierLoss(nn.Module):  # nn.Module
    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super(FourierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, sr, hr):
        sr_w, hr_w = self.addWindows(sr, hr)
        sr_mag, sr_ang = self.comFourier(sr_w)
        hr_mag, hr_ang = self.comFourier(hr_w)
        mag_loss = self.get_l1loss(sr_mag, hr_mag, weight=self.loss_weight, reduction=self.reduction)
        ang_loss = self.get_angleloss(sr_ang, hr_ang, weight=self.loss_weight, reduction=self.reduction)
        # ang_loss = self.get_l1loss(sr_ang, hr_ang, weight=self.loss_weight, reduction=self.reduction)
        return (mag_loss + ang_loss)

    def comFourier(self, image):
        frm_list = []
        fre_quen = []
        for i in range(3):
            in_ = image[:, i:i + 1, :, :]
            fftn = fft.fftn(in_, dim=(2, 3))
            # add shift
            fftn_shift = fft.fftshift(fftn)  # + 1e-8
            # print('fftn:', fftn_shift.size())
            frm_list.append(fftn_shift.real)
            fre_quen.append(fftn_shift.imag)
        fre_mag = torch.cat(frm_list, dim=1)
        fre_ang = torch.cat(fre_quen, dim=1)

        return fre_mag, fre_ang

    def addWindows(self, sr, hr):
        b, c, h, w = sr.size()
        win1 = torch.hann_window(h).reshape(h, 1)
        win2 = torch.hann_window(w).reshape(1, w)
        win = torch.mm(win1, win2)
        sr, hr = sr * win, hr * win
        return sr, hr

    def get_angleloss(self, pred, target, weight, reduction):
        # minimum = torch.minimum(pred, target)
        diff = pred - target
        minimum = torch.min(diff)
        loss = torch.mean(torch.abs(minimum))
        return weight * loss

    def get_l1loss(self, pred, target, weight, reduction):
        loss = F.l1_loss(pred, target, reduction=reduction)
        return weight * loss

if __name__ == '__main__':
    sr_tensor = torch.rand([1, 3, 256, 256])
    hr_tensor = torch.rand([1, 3, 256, 256])
    F_loss = FourierLoss()
    f_loss = F_loss.forward(sr_tensor, hr_tensor)
    print('f_Loss:', f_loss)