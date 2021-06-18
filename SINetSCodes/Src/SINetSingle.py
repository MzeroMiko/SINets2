import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
from torchvision.models import resnet50
from .Res2Net_v1b import res2net50_v1b_26w_4s

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


class Res_Features(nn.Module):
    def __init__(self):
        super(Res_Features, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

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
            BasicConv2d(in_channel, in_channel, 3, padding=1),
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


class Single_Decs(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Single_Decs, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        # self.dec3 = Dec(1024, 3*channel, 1024//4)
        self.dec2 = Dec(512, 2*channel, 512//4)
        self.dec1 = Dec(256, channel, 256//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, x1_sm, x2_sm, x3_sm, x4_sm = self.resf(x)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        x3_im = camouflage_map_sm
        x2_im = self.dec2(x2_sm, x3_im)
        x1_im = self.dec1(x1_sm, self.upsample_2(x2_im))

        return (
            self.upsample_8(x3_im),
            self.upsample_8(x2_im),
            self.upsample_4(x1_im),
        ) 


class Single_MinDecs2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Single_MinDecs2, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(512, channel)
        self.rf3_sm = RF(1024, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.dec2 = Dec(512, 2*channel, 512//4)
        self.dec1 = Dec(256, channel, 256//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, x1_sm, x2_sm, x3_sm, x4_sm = self.resf(x)

        x2_sm_rf = self.rf2_sm(x2_sm)
        x3_sm_rf = self.rf3_sm(x3_sm)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        x3_im = camouflage_map_sm
        x2_im = self.dec2(x2_sm, x3_im)
        x1_im = self.dec1(x1_sm, self.upsample_2(x2_im))

        return (
            self.upsample_8(x3_im),
            self.upsample_8(x2_im),
            self.upsample_4(x1_im),
        ) 


class Single_MinDec2s2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Single_MinDec2s2, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(512, channel)
        self.rf3_sm = RF(1024, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.dec2 = Dec2(512, 2*channel, 512//4)
        self.dec1 = Dec2(256, channel, 256//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, x1_sm, x2_sm, x3_sm, x4_sm = self.resf(x)

        x2_sm_rf = self.rf2_sm(x2_sm)
        x3_sm_rf = self.rf3_sm(x3_sm)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        x3_im = camouflage_map_sm
        x2_im = self.dec2(x2_sm, x3_im)
        x1_im = self.dec1(x1_sm, self.upsample_2(x2_im))

        return (
            self.upsample_8(x3_im),
            self.upsample_8(x2_im),
            self.upsample_4(x1_im),
        ) 


class Single_MTune_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Single_MTune_Dec, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune(channel, channel, scale_factor=1)
        self.mt3 = MTune(channel, channel, scale_factor=2)
        self.mt4 = MTune(channel, channel, scale_factor=4)
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)
        
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, x1, x2_sm, x3_sm, x4_sm = self.resf(x)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)
        tag = camouflage_map_sm.sigmoid()

        x2_im_rf = self.mt2(x2_sm_rf, tag)
        x3_im_rf = self.mt3(x3_sm_rf, tag)
        x4_im_rf = self.mt4(x4_sm_rf, tag)
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        out1 = self.dec1(x1, self.upsample_2(camouflage_map_im))

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im), self.upsample_4(out1)


class Single_MTune_MinDec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Single_MTune_MinDec, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(512, channel)
        self.rf3_sm = RF(1024, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune(channel, channel, scale_factor=1)
        self.mt3 = MTune(channel, channel, scale_factor=2)
        self.mt4 = MTune(channel, channel, scale_factor=4)
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)
        
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, x1, x2_sm, x3_sm, x4_sm = self.resf(x)

        # x2_sm_cat = torch.cat((x2_sm,
        #                        self.upsample_2(x3_sm),
        #                        self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        # x3_sm_cat = torch.cat((x3_sm,
        #                        self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm)
        x3_sm_rf = self.rf3_sm(x3_sm)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)
        tag = camouflage_map_sm.sigmoid()

        x2_im_rf = self.mt2(x2_sm_rf, tag)
        x3_im_rf = self.mt3(x3_sm_rf, tag)
        x4_im_rf = self.mt4(x4_sm_rf, tag)
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        out1 = self.dec1(x1, self.upsample_2(camouflage_map_im))

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im), self.upsample_4(out1)


# --------------------
# Re: Try To Minimize the Parameter

class Re_Single_MTune_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Re_Single_MTune_Dec, self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune(channel, channel, scale_factor=1)
        self.mt3 = MTune(channel, channel, scale_factor=2)
        self.mt4 = MTune(channel, channel, scale_factor=4)
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)
        
    def forward(self, x):
        
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2_sm = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3_sm = self.resnet.layer3(x2_sm)     # bs, 1024, 22, 22
        x4_sm = self.resnet.layer4(x3_sm)     # bs, 2048, 11, 11

        x2_sm_cat = torch.cat((x2_sm,
                F.interpolate(x3_sm, scale_factor=2, mode='bilinear', align_corners=True),
                F.interpolate(
                    F.interpolate(x4_sm, scale_factor=2, mode='bilinear', align_corners=True),
                    scale_factor=2, mode='bilinear', align_corners=True)
                ),dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                F.interpolate(x4_sm, scale_factor=2, mode='bilinear', align_corners=True),
                ), dim=1)  # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)
        tag = camouflage_map_sm.sigmoid()

        x2_im_rf = self.mt2(x2_sm_rf, tag)
        x3_im_rf = self.mt3(x3_sm_rf, tag)
        x4_im_rf = self.mt4(x4_sm_rf, tag)
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        out1 = self.dec1(x1, F.interpolate(camouflage_map_im, scale_factor=2, mode='bilinear', align_corners=True))

        return (
            F.interpolate(camouflage_map_sm, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(camouflage_map_im, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(out1, scale_factor=4, mode='bilinear', align_corners=True),
        )


class Re_Single_Decs(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Re_Single_Decs, self).__init__()

        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.dec2 = Dec(512, 2*channel, 512//4)
        self.dec1 = Dec(256, channel, 256//4)

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1_sm = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2_sm = self.resnet.layer2(x1_sm)     # bs, 512, 44, 44
        x3_sm = self.resnet.layer3(x2_sm)     # bs, 1024, 22, 22
        x4_sm = self.resnet.layer4(x3_sm)     # bs, 2048, 11, 11

        x2_sm_cat = torch.cat((x2_sm,
                F.interpolate(x3_sm, scale_factor=2, mode='bilinear', align_corners=True),
                F.interpolate(
                    F.interpolate(x4_sm, scale_factor=2, mode='bilinear', align_corners=True),
                    scale_factor=2, mode='bilinear', align_corners=True)
                ),dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                F.interpolate(x4_sm, scale_factor=2, mode='bilinear', align_corners=True),
                ), dim=1)  # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        x3_im = camouflage_map_sm
        x2_im = self.dec2(x2_sm, x3_im)
        x1_im = self.dec1(x1_sm, F.interpolate(x2_im, scale_factor=2, mode='bilinear', align_corners=True))

        return (
            F.interpolate(x3_im, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(x2_im, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(x1_im, scale_factor=4, mode='bilinear', align_corners=True),
            # self.upsample_8(x3_im),
            # self.upsample_8(x2_im),
            # self.upsample_4(x1_im),
        ) 


class Re_Single_MinDecs2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Re_Single_MinDecs2, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(512, channel)
        self.rf3_sm = RF(1024, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.dec2 = Dec(512, 2*channel, 512//4)
        self.dec1 = Dec(256, channel, 256//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, x1_sm, x2_sm, x3_sm, x4_sm = self.resf(x)

        x2_sm_rf = self.rf2_sm(x2_sm)
        x3_sm_rf = self.rf3_sm(x3_sm)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        x3_im = camouflage_map_sm
        x2_im = self.dec2(x2_sm, x3_im)
        x1_im = self.dec1(x1_sm, self.upsample_2(x2_im))

        return (
            self.upsample_8(x3_im),
            self.upsample_8(x2_im),
            self.upsample_4(x1_im),
        ) 


class Re_Single_MinDec2s2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(Re_Single_MinDec2s2, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(512, channel)
        self.rf3_sm = RF(1024, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.dec2 = Dec2(512, 2*channel, 512//4)
        self.dec1 = Dec2(256, channel, 256//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, x1_sm, x2_sm, x3_sm, x4_sm = self.resf(x)

        x2_sm_rf = self.rf2_sm(x2_sm)
        x3_sm_rf = self.rf3_sm(x3_sm)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        x3_im = camouflage_map_sm
        x2_im = self.dec2(x2_sm, x3_im)
        x1_im = self.dec1(x1_sm, self.upsample_2(x2_im))

        return (
            self.upsample_8(x3_im),
            self.upsample_8(x2_im),
            self.upsample_4(x1_im),
        ) 


# SINet v2 --------------------------------------------------------------------------


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
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
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
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
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


# Group-Reversal Attention (GRA) Block
class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y


class Network__(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network__, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = resnet50(pretrained=imagenet_pretrained)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)

        # # ---- reverse stage ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x):
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        # Receptive Field Block (enhanced)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')    # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred


# 12.244G 24.927M 26.976M
class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)

        # # ---- reverse stage ----
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)

    def forward(self, x):
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44
        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11

        # Receptive Field Block (enhanced)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')    # Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear')
        ra4_feat = self.RS5(x4_rfb, guidance_g)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear')
        ra3_feat = self.RS4(x3_rfb, guidance_5)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear')
        ra2_feat = self.RS3(x2_rfb, guidance_4)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred


if __name__ == '__main__':
    import numpy as np
    from time import time
    from Res2Net_v1b import res2net50_v1b_26w_4s as resnet50
    Network = Single_MTune_Dec
    net = Network().cuda()
    net.eval()

    dump_x = torch.randn(1, 3, 352, 352).cuda()
    frame_rate = np.zeros((1000, 1))
    for i in range(1000):
        torch.cuda.synchronize()
        start = time()
        y = net(dump_x)
        torch.cuda.synchronize()
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        print(i, '->', running_frame_rate)
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
