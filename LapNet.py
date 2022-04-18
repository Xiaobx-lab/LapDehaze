import time

import torch.nn as nn
import torch.nn.functional as F
import torch
from Moudle.unet_model import UNet

class Lap_Pyramid_Bicubic(nn.Module):
    """

    """
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Bicubic, self).__init__()

        self.interpolate_mode = 'bicubic'
        self.num_high = num_high

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for i in range(self.num_high):
            down = nn.functional.interpolate(current, size=(current.shape[2] // 2, current.shape[3] // 2), mode=self.interpolate_mode, align_corners=True)
            up = nn.functional.interpolate(down, size=(current.shape[2], current.shape[3]), mode=self.interpolate_mode, align_corners=True)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[-1]
        for level in reversed(pyr[:-1]):
            image = F.interpolate(image, size=(level.shape[2], level.shape[3]), mode=self.interpolate_mode, align_corners=True) + level
        return image

class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda'), channels=3):  # 高斯核
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):  # 高斯卷积
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):  # 拉普拉斯金字塔高频分量和低频分量的区分
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up # 高频残差
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):  # list
        image = pyr[-1]
        for level in reversed(pyr[:-1]):  # 反向遍历除了最后一个元素
            up = self.upsample(image)
            if up.shape[2] != level.shape[2] or up.shape[3] != level.shape[3]:
                up = nn.functional.interpolate(up, size=(level.shape[2], level.shape[3]))
            image = up + level
        return image


class Trans_high(nn.Module):
    def __init__(self,  num_high=3):
        super(Trans_high, self).__init__()

        self.num_high = num_high
        model = [nn.Conv2d(9, 64, 3, padding=1)]
        model += [UNet(64,64)]

        model += [nn.Conv2d(64, 3, 3, padding=1)]

        self.model = nn.Sequential(*model)

        for i in range(self.num_high):
            trans_mask_block = nn.Sequential(
                nn.Conv2d(3, 16, 1),
                nn.LeakyReLU(),
                nn.Conv2d(16, 3, 1))
            setattr(self, 'trans_mask_block_{}'.format(str(i)), trans_mask_block)

    def forward(self, x, pyr_original, fake_low):   # concat分量 lp分量list 低频处理分量

        pyr_result = []
        mask = self.model(x)  # 算一个掩码出来
        for i in range(self.num_high):
            mask = nn.functional.interpolate(mask, size=(pyr_original[-2-i].shape[2], pyr_original[-2-i].shape[3]))
            result_highfreq = torch.mul(pyr_original[-2-i], mask) + pyr_original[-2-i]
            self.trans_mask_block = getattr(self, 'trans_mask_block_{}'.format(str(i)))
            result_highfreq = self.trans_mask_block(result_highfreq)
            setattr(self, 'result_highfreq_{}'.format(str(i)), result_highfreq)

        for i in reversed(range(self.num_high)):
            result_highfreq = getattr(self, 'result_highfreq_{}'.format(str(i)))
            pyr_result.append(result_highfreq)

        pyr_result.append(fake_low)  # 低频分量追加到后面即可

        return pyr_result

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class LapNet(nn.Module):
    def __init__(self, num_high=3):
        super(LapNet, self).__init__()
        self.lap_pyramid = Lap_Pyramid_Conv(num_high)
        trans_high = Trans_high(num_high=num_high)
        self.trans_high = trans_high.cuda()
        self.unet = UNet(3,3).cuda()
        self.sig = nn.Sigmoid().cuda()
        self.relu = nn.ReLU().cuda()
    def forward(self, real_A_full):

        pyr_A = self.lap_pyramid.pyramid_decom(img=real_A_full)
        fake_B_low = self.unet(pyr_A[-1])
        fake_B_low = self.relu((fake_B_low*pyr_A[-1])-fake_B_low+1)
        real_A_up = nn.functional.interpolate(pyr_A[-1], size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        fake_B_up = nn.functional.interpolate(fake_B_low, size=(pyr_A[-2].shape[2], pyr_A[-2].shape[3]))
        high_with_low = torch.cat([pyr_A[-2], real_A_up, fake_B_up], 1)
        pyr_A_trans = self.trans_high(high_with_low, pyr_A, fake_B_low)  # list concat分量 lp分量list 低频处理分量
        fake_B_full = self.lap_pyramid.pyramid_recons(pyr_A_trans)
        fake_B_full = self.sig(fake_B_full)
        return fake_B_full
if __name__ == "__main__":
    device = torch.device("cuda")
    X = torch.Tensor(1,3,2160,3840).to(device)
    net = LapNet(num_high=4)
    y = net(X)
