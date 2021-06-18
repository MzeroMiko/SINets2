import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
import torchvision.models as models
from torch.nn.parameter import Parameter
from torchvision.models import resnet50
from .ResNet import ResNet_2Branch
from .complexPyTorch.complexLayers import ComplexConv2d, ComplexBatchNorm2d, ComplexReLU

" -------------------- SA ------------------------------------------"

def _get_kernel(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    """
        normalization
    :param: in_
    :return:
    """
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_ - min_ + 1e-8)


class SA(nn.Module):
    """
        holistic attention src
    """
    def __init__(self):
        super(SA, self).__init__()
        gaussian_kernel = np.float32(_get_kernel(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention, x, visual=False):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)       # normalization
        x = torch.mul(x, soft_attention.max(attention))     # mul
        if visual:
            return x, soft_attention.max(attention)
        else:
            return x


class GA(nn.Module):
    def __init__(self):
        super(GA, self).__init__()
        gaussian_kernel = np.float32(_get_kernel(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel))

    def forward(self, attention):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)       # normalization
        return soft_attention.max(attention)


" -------------------- Basic ------------------------------------------"

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class RF(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel):
        super(RF, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class PDC_SM(nn.Module):
    # Partial Decoder Component (Search Module)
    def __init__(self, channel):
        super(PDC_SM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv4 = BasicConv2d(4*channel, 4*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(4*channel, 1, 1)

    def forward(self, x1, x2, x3, x4):
        # print x1.shape, x2.shape, x3.shape, x4.shape
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2)), x4), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class PDC_IM(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(PDC_IM, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

" -------------------- PDC Series ------------------------------------------"

class ChannelAttention(nn.Module):
    # https://github.com/luuuyi/CBAM.PyTorch/tree/master/
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return res * self.sigmoid(out)


class SpatialAttention(nn.Module):
    # https://github.com/luuuyi/CBAM.PyTorch/tree/master/
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return res * self.sigmoid(x)


class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.chat = ChannelAttention(in_planes=in_channels) 
        self.spat = SpatialAttention()
        # self.conv = BasicConv2d()

    def forward(self, x):
        x = self.chat(x) + self.spat(x)
        return x


class Grid_Attention(nn.Module):
    def __init__(self, Fg, Fl, Fint):
        super(Grid_Attention, self).__init__()

        self.Wf = nn.Sequential(
            nn.Conv2d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(Fint)
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(Fint)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(Fint, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        size = x.size()[2:]
        if g.size()[2:] != size:
            g = nn.functional.interpolate(g, size=size, mode='bilinear', align_corners=True)
        psi = self.psi(self.Wg(g) + self.Wf(x))
        return x * psi


class PDC_IMATT(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(PDC_IMATT, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)

        self.attend = Attention(3*channel)

        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)
        
        x3_2 = self.attend(x3_2)
        
        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class PDCN(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(PDCN, self).__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
        )
        self.cat1 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.cat2 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv_final = nn.Sequential(
            BasicConv2d(3*channel, 3*channel, 3, padding=1),
            nn.Conv2d(3*channel, 1, 1),
        )

    def forward(self, *inps):
        x1, x2, x3 = inps[0], inps[1], inps[2]  
        x1_1 = x1
        x2_1 = x2 * self.up1(x1)
        x3_1 = x3 * self.up2(x2) * self.up3(x2_1)

        x = self.cat1(torch.cat([x2_1, self.up4(x1_1)], dim=1))
        x = self.cat2(torch.cat([x3_1, self.up5(x)], dim=1))

        x = self.conv_final(x)

        return x


class PDCNT(nn.Module):
    def __init__(self, channel):
        super(PDCNT, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample2 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample3 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample4 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )
        self.conv_upsample5 = nn.Sequential(
            nn.ConvTranspose2d(2*channel, 2*channel, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(2*channel),
            nn.ReLU(inplace=True),
        )

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(x1) * x2
        x3_1 = self.conv_upsample2(x2_1) * self.conv_upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class PDCA(nn.Module):
    def __init__(self, channel):
        super(PDCA, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.att1 = Grid_Attention(channel, channel, channel//2)
        self.att2 = Grid_Attention(channel, channel, channel//2)
        self.att3 = Grid_Attention(channel, channel, channel//2)
        self.conv_final = nn.Sequential(
            BasicConv2d(4*channel, 4*channel, 3, padding=1),
            BasicConv2d(4*channel, 4*channel, 3, padding=1),
            nn.Conv2d(4*channel, 1, 1),
        ) 

    def forward(self, x1, x2, x3):
        x1 = self.conv_upsample1(self.upsample(x1))
        x1_1 = self.conv_upsample3(self.upsample(x1))

        x2 = self.att1(x1, x2)
        x2_1 = self.conv_upsample2(self.upsample(x2))

        x3_1 = self.att2(x1_1, x3)
        x4_1 = self.att3(x2_1, x3)
        
        x3 = self.conv_final(torch.cat((x1_1, x2_1, x3_1, x4_1), dim=1))
        return x3


"---------------------------- RF Series ---------------------------- "

class DCT1d(nn.Linear):
    def __init__(self, in_features, idct=False, norm='ortho', bias=False):
        trans = self.dct if idct == False else self.idct 
        self.weight_data = trans(torch.eye(in_features), norm=norm).data.t()
        super(DCT1d, self).__init__(in_features, in_features, bias=bias)
       
    @staticmethod
    def _rfft(x, signal_ndim=1, onesided=True):
        # b = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
        # b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
        # torch 1.8.0 torch.fft.rfft  to torch 1.5.0 torch.rfft as signal_ndim=1
        # written by mzero
        odd_shape1 = (x.shape[1] % 2 != 0)
        x = torch.fft.rfft(x)
        x = torch.cat([x.real.unsqueeze(dim=2), x.imag.unsqueeze(dim=2)], dim=2)
        if onesided == False:
            _x = x[:, 1:, :].flip(dims=[1]).clone() if odd_shape1 else x[:, 1:-1, :].flip(dims=[1]).clone()
            _x[:,:,1] = -1 * _x[:,:,1]
            x = torch.cat([x, _x], dim=1)
        return x

    @staticmethod
    def _irfft(x, signal_ndim=1, onesided=True):
        # b = torch.Tensor([[1,2,3,4,5],[2,3,4,5,6]])
        # b = torch.Tensor([[1,2,3,4,5,6],[2,3,4,5,6,7]])
        # torch 1.8.0 torch.fft.irfft  to torch 1.5.0 torch.irfft as signal_ndim=1
        # written by mzero
        if onesided == False:
            res_shape1 = x.shape[1]
            x = x[:,:(x.shape[1] // 2 + 1),:]
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x, n=res_shape1)
        else:
            x = torch.complex(x[:,:,0].float(), x[:,:,1].float())
            x = torch.fft.irfft(x)
        return x

    @classmethod
    def dct(cls, x, norm=None):
        # https://github.com/zh217/torch-dct
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        # Vc = torch.fft.rfft(v, 1, onesided=False)
        Vc = cls._rfft(v, 1, onesided=False)

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)

        return V

    @classmethod
    def idct(cls, X, norm=None):
        # https://github.com/zh217/torch-dct
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        # v = torch.fft.irfft(V, 1, onesided=False)
        v = cls._irfft(V, 1, onesided=False)

        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]

        return x.view(*x_shape)

    def reset_parameters(self):
        self.weight.data = self.weight_data
        self.weight.requires_grad = False


class DCT2d(nn.Module):
    def __init__(self, in_features, idct=False, norm='ortho'):
        super(DCT2d, self).__init__()
        self.dct1d = DCT1d(in_features, idct=idct, norm=norm)

    def forward(self, x):
        X1 = self.dct1d(x)
        X2 = self.dct1d(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)


class DCT_Conv(nn.Module):
    def __init__(self, in_channel=32, squeeze_channel=8, resolution=44, conv_final=False):
        super(DCT_Conv, self).__init__()
        self.dct2d = DCT2d(resolution, idct=False, norm='ortho')
        self.idct2d = DCT2d(resolution, idct=True, norm='ortho')
        # https://github.com/zh217/torch-dct
        # self.dct_layer = LinearDCT(in_features=resolution, type='dct', norm='ortho') 
        # self.idct_layer = LinearDCT(in_features=resolution, type='idct', norm='ortho')
        self.se = nn.Sequential(
            BasicConv2d(in_channel, squeeze_channel, 1),
            BasicConv2d(squeeze_channel, in_channel, 1),
        )
        self.conv1 = BasicConv2d(in_channel, in_channel, 3, padding=1)
        self.conv2 = BasicConv2d(2*in_channel, in_channel, 3, padding=1) if conv_final else None

    def forward(self, x):
        # x_dct = apply_linear_2d(x, self.dct_layer)
        # x_idct = apply_linear_2d(self.se(x_dct) + x_dct, self.idct_layer)
        x_dct = self.dct2d(x)
        x_idct = self.idct2d(self.se(x_dct) + x_dct)
        x = torch.cat((self.conv1(x_idct) + x, x), dim=1)
        return self.conv2(x) if self.conv2 is not None else x   


class FFT_Conv(nn.Module):
    def __init__(self, in_channel=32, squeeze_channel=8, resolution=44, conv_final=False):
        super(FFT_Conv, self).__init__()
        self.se = nn.Sequential(
            ComplexConv2d(in_channel, squeeze_channel, 1),
            ComplexBatchNorm2d(squeeze_channel),
            ComplexReLU(),
            ComplexConv2d(squeeze_channel, in_channel, 1),
            ComplexBatchNorm2d(in_channel),
            ComplexReLU(),
        )
        self.conv1 = BasicConv2d(in_channel, in_channel, 3, padding=1)
        self.conv2 = BasicConv2d(2*in_channel, in_channel, 3, padding=1) if conv_final else None
        self.resolution = resolution

    def forward(self, x):
        x_fft = torch.fft.rfftn(x.float(), dim=(2,3))
        x_se = self.se(x_fft)
        x_ifft = torch.fft.irfftn(x_se + x_fft, s=x.size()[2:]).type_as(x)
        x = torch.cat((self.conv1(x_ifft) + x, x), dim=1)
        return self.conv2(x) if self.conv2 is not None else x   


class DCTATT_Conv(nn.Module):
    def __init__(self, in_channel=32, squeeze_channel=8, resolution=44, conv_final=False):
        super(DCTATT_Conv, self).__init__()
        self.dct2d = DCT2d(resolution, idct=False, norm='ortho')
        self.idct2d = DCT2d(resolution, idct=True, norm='ortho')
        self.att = Attention(in_channel)
        self.conv1 = BasicConv2d(in_channel, in_channel, 3, padding=1)
        self.conv2 = BasicConv2d(2*in_channel, in_channel, 3, padding=1) if conv_final else None

    def forward(self, x):
        x_dct = self.dct2d(x)
        x_idct = self.idct2d(self.att(x_dct) + x_dct)
        x = torch.cat((self.conv1(x_idct) + x, x), dim=1)
        return self.conv2(x) if self.conv2 is not None else x   


class RF2(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel, dilations=[1,3,5,7]):
        super(RF2, self).__init__()
        self.branches = nn.ModuleList([])
        for i in range(len(dilations)):
            self.branches.append(self.__make_dialation(in_channel + i * out_channel, out_channel, dilations[i]))
        self.conv_cat = BasicConv2d(in_channel + len(dilations)*out_channel, out_channel, 3, padding=1)

    def __make_dialation(self, in_channel, out_channel, dilation):
        return nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=dilation, padding=dilation//2),
            BasicConv2d(out_channel, out_channel, 3, padding=dilation, dilation=dilation),
        )

    def forward(self, x):
        xs = [x]
        for i in range(len(self.branches)):
            xs.append(self.branches[i](torch.cat(xs, dim=1)))
        return self.conv_cat(torch.cat(xs, dim=1))


class RF3(nn.Module):
    # Revised from: Receptive Field Block Net for Accurate and Fast Object Detection, 2018, ECCV
    # GitHub: https://github.com/ruinmessi/RFBNet
    def __init__(self, in_channel, out_channel, dilations=[1,3,5,7]):
        super(RF3, self).__init__()
        self.branches = nn.ModuleList([])
        for i in range(len(dilations)):
            self.branches.append(self.__make_dialation(in_channel + i *out_channel, out_channel, dilations[i]))

        self.conv_cat = BasicConv2d(len(dilations)*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
        self.relu = nn.ReLU(inplace=True)

    def __make_dialation(self, in_channel, out_channel, dilation):
        if dilation == 1:
            return BasicConv2d(in_channel, out_channel, 1)
        else:
            return nn.Sequential(
                BasicConv2d(in_channel, out_channel, 1),
                BasicConv2d(out_channel, out_channel, kernel_size=(1, dilation), padding=(0, dilation//2)),
                BasicConv2d(out_channel, out_channel, kernel_size=(dilation, 1), padding=(dilation//2, 0)),
                BasicConv2d(out_channel, out_channel, 3, padding=dilation, dilation=dilation),
            )

    def forward(self, x):
        xs = [x]
        for i in range(len(self.branches)):
            xs.append(self.branches[i](torch.cat(xs, dim=1)))
        return self.relu(self.conv_cat(torch.cat(xs[1:], dim=1)) + self.conv_res(x))


" --------------------------------- Dec Series ---------------------- "

class Dec(nn.Module):
    def __init__(self, in_channel, channel, tag_channel=0):
        super(Dec, self).__init__()
        self.tag_channel = tag_channel if tag_channel != 0 else in_channel
        self.half_channel = channel // 2
        self.conv = BasicConv2d(in_channel + self.tag_channel, in_channel//2, 3, padding=1)
        self.rf_cat = RF(in_channel // 2, channel)
        self.conv_final = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1),
        )

    def forward(self, x, tag):
        x = torch.cat([x,tag.repeat(1,self.tag_channel,1,1)], dim=1)
        x = self.rf_cat(self.conv(x))
        x = torch.cat([x, x * tag.sigmoid()], dim=1)
        x = self.conv_final(x) + tag
        return x


class Dec2(nn.Module):
    def __init__(self, in_channel, channel, tag_channel=0):
        super(Dec2, self).__init__()
        self.tag_channel = tag_channel if tag_channel != 0 else in_channel
        self.half_channel = channel // 2
        self.conv = nn.Sequential(
            BasicConv2d(in_channel + self.tag_channel, in_channel, 3, padding=1),
            BasicConv2d(in_channel, channel, 3, padding=1),
        ) 
        self.conv_final = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1),
        )

    def forward(self, x, tag):
        x = torch.cat([x,tag.repeat(1,self.tag_channel,1,1)], dim=1)
        x = self.conv(x)
        x = torch.cat([x, x * tag.sigmoid()], dim=1)
        x = self.conv_final(x) + tag
        return x


class Dec3(nn.Module):
    def __init__(self, in_channel, channel, tag_channel=0):
        super(Dec3, self).__init__()
        self.tag_channel = tag_channel if tag_channel != 0 else in_channel
        self.half_channel = channel // 2
        self.conv_cat = BasicConv2d(in_channel + tag_channel, in_channel//2, 3, padding=1)
        self.att = Attention(in_channel)
        self.conv_final = nn.Sequential(
            BasicConv2d(in_channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1),
        )

    def forward(self, x, tag):
        x = torch.cat([x,tag.repeat(1,self.tag_channel,1,1)], dim=1)
        x = self.conv_cat(x)
        x = torch.cat([x, x * tag.sigmoid()], dim=1)
        x = self.att(x)
        x = self.conv_final(x) + tag
        return x


class Dec4(nn.Module):
    def __init__(self, in_channel, channel, tag_channel=0):
        super(Dec4, self).__init__()
        self.tag_channel = tag_channel if tag_channel != 0 else in_channel
        self.conv_cat = nn.Sequential(
            BasicConv2d(in_channel + self.tag_channel, in_channel + self.tag_channel, 3, padding=1),
            BasicConv2d(in_channel + self.tag_channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1),
        )

    def forward(self, x, tag):
        x = torch.cat([x, (1 - tag.sigmoid()).repeat(1,self.tag_channel,1,1)], dim=1)
        x = self.conv_cat(x) + tag
        return x


" ------------------------------- Single Series ---------------------- "


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class FineTune(nn.Module):
    def __init__(self, in_channels, channel, tag_channel=0, scale_factor=2):
        super(FineTune, self).__init__()
        self.tag_channel = in_channels if tag_channel == 0 else tag_channel
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(channel, channel, 3, padding=1),
        ) 
        self.up = self._make_up(scale_factor)
        self.layers = self._make_layer(in_channels + self.tag_channel, in_channels + self.tag_channel, 4)
        self.conv1 = BasicConv2d(in_channels + self.tag_channel, channel, 3, padding=1)
        self.conv2 = nn.Conv2d(channel, 1, 1)

    def _make_layer(self, in_planes, out_planes, blocks):
        block = Bottleneck
        planes = out_planes // block.expansion
        layers = []
        layers.append(block(in_planes, planes))
        for _ in range(1, blocks):
            layers.append(block(out_planes, planes))
        return nn.Sequential(*layers)

    def _make_up(self, scale_factor):
        layers = []
        while scale_factor != 1:
            scale_factor = scale_factor / 2
            layers.append(self.upsample2)
        return nn.Sequential(*layers)

    def forward(self, x, tag):
        tag = tag.repeat(1, self.tag_channel, 1, 1)
        x = self.up(x)
        x = torch.cat([x, tag], dim=1)
        x = self.layers(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FineTune1(nn.Module):
    def __init__(self, in_channels, channel, tag_channel=0, scale_factor=2):
        super(FineTune1, self).__init__()
        self.tag_channel = in_channels if tag_channel == 0 else tag_channel
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(channel, channel, 3, padding=1),
        ) 
        self.up = self._make_up(scale_factor)
        self.layers = self._make_layer(2*in_channels + self.tag_channel, 2*in_channels + self.tag_channel, 4)
        self.conv1 = BasicConv2d(2*in_channels + self.tag_channel, channel, 3, padding=1)
        self.conv2 = nn.Conv2d(channel, 1, 1)

    def _make_layer(self, in_planes, out_planes, blocks):
        block = Bottleneck
        planes = out_planes // block.expansion
        layers = []
        layers.append(block(in_planes, planes))
        for _ in range(1, blocks):
            layers.append(block(out_planes, planes))
        return nn.Sequential(*layers)

    def _make_up(self, scale_factor):
        layers = []
        while scale_factor != 1:
            scale_factor = scale_factor / 2
            layers.append(self.upsample2)
        return nn.Sequential(*layers)

    def forward(self, x, tag):
        tags = tag.repeat(1, self.tag_channel, 1, 1)
        x = self.up(x)
        x1 = x * (1-tag.sigmoid()) 
        x = torch.cat([x, x1, tags], dim=1)
        x = self.layers(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class FineTune2(nn.Module):
    def __init__(self, in_channels, channel, tag_channel=0, scale_factor=2):
        super(FineTune2, self).__init__()
        self.tag_channel = in_channels if tag_channel == 0 else tag_channel
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(channel, channel, 3, padding=1),
        ) 
        self.up = self._make_up(scale_factor)
        self.conv = nn.Sequential(
            BasicConv2d(in_channels + self.tag_channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1),
        )

    def _make_up(self, scale_factor):
        layers = []
        while scale_factor != 1:
            scale_factor = scale_factor / 2
            layers.append(self.upsample2)
        return nn.Sequential(*layers)

    def forward(self, x, tag):
        x = torch.cat([self.up(x), tag.repeat(1, self.tag_channel, 1, 1)], dim=1)
        return self.conv(x)


class FineTune3(nn.Module):
    def __init__(self, in_channels, channel, tag_channel=0, scale_factor=2):
        super(FineTune3, self).__init__()
        self.tag_channel = in_channels if tag_channel == 0 else tag_channel
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(channel, channel, 3, padding=1),
        ) 
        self.up = self._make_up(scale_factor)
        self.conv1 = nn.Sequential(
            BasicConv2d(in_channels + self.tag_channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.conv2 = nn.Sequential(
            BasicConv2d(channel + self.tag_channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )
        self.conv3 = nn.Conv2d(channel, 1, 1)

    def _make_up(self, scale_factor):
        layers = []
        while scale_factor != 1:
            scale_factor = scale_factor / 2
            layers.append(self.upsample2)
        return nn.Sequential(*layers)

    def forward(self, x, tag):
        x = torch.cat([self.up(x), tag.repeat(1, self.tag_channel, 1, 1)], dim=1)
        x = torch.cat([self.conv1(x), tag.repeat(1, self.tag_channel, 1, 1)], dim=1)
        return self.conv3(self.conv2(x))


" ------------------------------ Single Series 2 ------------------------ "

class MTune(nn.Module):
    def __init__(self, in_channel, channel, tag_channel=0, scale_factor=2):
        super(MTune, self).__init__()
        self.tag_channel = in_channel if tag_channel == 0 else tag_channel

        if scale_factor == 1:
            self.downSample = nn.Identity()
        else:
            self.downSample = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)

        self.cat_conv = nn.Sequential(
            BasicConv2d(in_channel + self.tag_channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )

        self.conv_final = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1),
        )

    def forward(self, x, tag):
        x = torch.cat([x, self.downSample(tag).repeat(1, self.tag_channel, 1, 1)], dim=1)
        x = self.cat_conv(x)
        x = x + self.conv_final(x)
        return x


class MTune2(nn.Module):
    def __init__(self, in_channel, channel, tag_channel=0, scale_factor=2):
        super(MTune2, self).__init__()
        self.tag_channel = in_channel if tag_channel == 0 else tag_channel
        self.channel = channel

        if scale_factor == 1:
            self.downSample = nn.Identity()
        else:
            self.downSample = nn.MaxPool2d(kernel_size=scale_factor, stride=scale_factor)

        self.cat_conv = nn.Sequential(
            BasicConv2d(in_channel + self.tag_channel, in_channel, 3, padding=1),
            BasicConv2d(in_channel, 2*channel, 3, padding=1),
        )

        self.conv_final = nn.Sequential(
            BasicConv2d(3*channel, 3*channel, 3, padding=1),
            nn.Conv2d(3*channel, channel, 3, padding=1),
        )

    def forward(self, x, tag):
        x = torch.cat([x, self.downSample(tag).repeat(1, self.tag_channel, 1, 1)], dim=1)
        x = self.cat_conv(x) 
        x = torch.cat([x, self.downSample(tag.sigmoid()).repeat(1, self.channel, 1, 1)], dim=1)
        x = self.conv_final(x)
        return x


class Res_Features(nn.Module):
    def __init__(self):
        super(Res_Features, self).__init__()
        self.resnet = resnet50(pretrained=True)

    def forward(self, x):
        c = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        c.append(x) # 64, /2
        x = self.resnet.maxpool(x)
        c.append(x) # 64, /4
        x = self.resnet.layer1(x)
        c.append(x) # 256, /4
        x = self.resnet.layer2(x)
        c.append(x) # 512, /8
        x = self.resnet.layer3(x)
        c.append(x) # 1024, /16
        x = self.resnet.layer4(x)
        c.append(x) # 2048, /32
        return tuple(c)



