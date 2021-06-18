import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from .ResNet import ResNet_2Branch
from torchvision.models import resnet50
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.stats as st

"ONLY DO With VISUAL"


" ---------------------- Original ---------------------------------------------" 


from .SINet_Parts import  _get_kernel, min_max_norm, GA, BasicConv2d, RF


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

    def forward(self, x1, x2, x3, visual=False):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)

        if not visual:
            return x
        else:
            return x, x3_2


class SINet_ResNet50(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_ResNet50, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_SM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x0 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        x01_down = self.downSample(x01)         # (BS, 320, 44, 44)
        x01_sm_rf = self.rf_low_sm(x01_down)    # (BS, 32, 44, 44)

        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa, x2_sa_att = self.SA(camouflage_map_sm.sigmoid(), x2, visual=True)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return (
            self.upsample_8(camouflage_map_sm), 
            self.upsample_8(camouflage_map_im),
            self.upsample_8(x2_sa_att),
            self.upsample_8(torch.mean(x01_sm_rf,dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(x2_sm_rf,dim=1).unsqueeze(dim=1)),
            self.upsample_8(self.upsample_2(torch.mean(x3_sm_rf,dim=1).unsqueeze(dim=1))),
            # self.upsample_8(torch.mean(x1,dim=1).unsqueeze(dim=1)),
            # self.upsample_8(torch.mean(x2,dim=1).unsqueeze(dim=1)),
        ) 
            
    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')


class SINet_Simp(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x0 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm, map_sm_x3_2 = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, visual=True)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa, x2_sa_att = self.SA(camouflage_map_sm.sigmoid(), x2, visual=True)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im, map_im_x3_2 = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf, visual=True)

        # ---- output ----
        return (
            # self.upsample_8(camouflage_map_sm), 
            # self.upsample_8(camouflage_map_im),
            # self.upsample_8(x2_sa_att),
            # self.upsample_8(torch.mean(x2_sm_rf,dim=1).unsqueeze(dim=1)),
            # self.upsample_8(self.upsample_2(torch.mean(x3_sm_rf,dim=1).unsqueeze(dim=1))),
            # self.upsample_8(self.upsample_2(self.upsample_2(torch.mean(x4_sm_rf,dim=1).unsqueeze(dim=1)))),
            # self.upsample_8(torch.mean(map_sm_x3_2,dim=1).unsqueeze(dim=1)),
            # self.upsample_8(torch.mean(map_im_x3_2,dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(x2_sa,dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(x2_im_rf,dim=1).unsqueeze(dim=1)),
        ) 

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')


" -------------------- PDC Series ------------------------------------------"


from .SINet_Parts import ChannelAttention, SpatialAttention, Attention, Grid_Attention


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

    def forward(self, x1, x2, x3, visual=False):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)

        x3_2 = self.conv_concat3(x3_2)
        
        x3_2_0 = x3_2
        x3_2 = self.attend(x3_2)
        
        x = self.conv4(x3_2)
        x = self.conv5(x)

        if visual:
            return x, x3_2_0, x3_2
        else:
            return x


class SINet_Simp_PDCATT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_PDCATT, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        # self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IMATT(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IMATT(channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x0 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        # x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        # x01_down = self.downSample(x01)         # (BS, 320, 44, 44)
        # x01_sm_rf = self.rf_low_sm(x01_down)    # (BS, 32, 44, 44)

        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm, map_sm_x3_2_0, map_sm_x3_2 = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, visual=True)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im, map_im_x3_2_0, map_im_x3_2 = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf, visual=True)

        # ---- output ----
        return (
            self.upsample_8(camouflage_map_sm), 
            self.upsample_8(camouflage_map_im),
            self.upsample_8(torch.mean(map_sm_x3_2_0, dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(map_sm_x3_2, dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(map_im_x3_2_0, dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(map_im_x3_2, dim=1).unsqueeze(dim=1)),
        )

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')


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

    def forward(self, x1, x2, x3, visual=False):
        x1 = self.conv_upsample1(self.upsample(x1))
        x1_1 = self.conv_upsample3(self.upsample(x1))

        x2 = self.att1(x1, x2)
        x2_1 = self.conv_upsample2(self.upsample(x2))

        x3_1 = self.att2(x1_1, x3)
        x4_1 = self.att3(x2_1, x3)
        
        x3 = self.conv_final(torch.cat((x1_1, x2_1, x3_1, x4_1), dim=1))
        
        if not visual:
            return x3
        else:
            # print(self.conv_final[0])
            x3_2 = self.conv_final[0](torch.cat((x1_1, x2_1, x3_1, x4_1), dim=1))
            return x3, x3_2


class SINet_Simp_PDCA(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_PDCA, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDCA(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDCA(channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x0 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        # x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        # x01_down = self.downSample(x01)         # (BS, 320, 44, 44)
        # x01_sm_rf = self.rf_low_sm(x01_down)    # (BS, 32, 44, 44)

        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm, map_sm_x3_2 = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, visual=True)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im, map_im_x3_2 = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf, visual=True)

        # ---- output ----
        return (
            self.upsample_8(camouflage_map_sm), 
            self.upsample_8(camouflage_map_im),
            self.upsample_8(torch.mean(map_sm_x3_2,dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(map_sm_x3_2,dim=1).unsqueeze(dim=1)),
        ) 

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')


" -------------------- SINet FREQ & Simp FREQ Series -----------------------------------------"


from .SINet_Parts import DCT1d, DCT2d, FFT_Conv, Res_Features


class DCT_Conv(nn.Module):
    def __init__(self, in_channel=32, squeeze_channel=8, resolution=44, conv_final=False, visual=False):
        super(DCT_Conv, self).__init__()
        self.visual = visual
        self.dct2d = DCT2d(resolution, idct=False, norm='ortho')
        self.idct2d = DCT2d(resolution, idct=True, norm='ortho')
        self.se = nn.Sequential(
            BasicConv2d(in_channel, squeeze_channel, 1),
            BasicConv2d(squeeze_channel, in_channel, 1),
        )
        self.conv1 = BasicConv2d(in_channel, in_channel, 3, padding=1)
        self.conv2 = BasicConv2d(2*in_channel, in_channel, 3, padding=1) if conv_final else None

    def forward(self, x, visual=False):
        visual = self.visual
        x_dct = self.dct2d(x)
        x_idct = self.idct2d(self.se(x_dct) + x_dct)
        x = torch.cat((self.conv1(x_idct) + x, x), dim=1)
        if not visual:
            return self.conv2(x) if self.conv2 is not None else x   
        else:
            return (
                self.conv2(x) if self.conv2 is not None else x,
                x_dct,
                self.se[0](x_dct),
                self.se(x_dct),
                x_idct,
            )   


class DCTATT_Conv(nn.Module):
    def __init__(self, in_channel=32, squeeze_channel=8, resolution=44, conv_final=False, visual=False):
        super(DCTATT_Conv, self).__init__()
        self.visual = visual
        self.dct2d = DCT2d(resolution, idct=False, norm='ortho')
        self.idct2d = DCT2d(resolution, idct=True, norm='ortho')
        self.att = Attention(in_channel)
        self.conv1 = BasicConv2d(in_channel, in_channel, 3, padding=1)
        self.conv2 = BasicConv2d(2*in_channel, in_channel, 3, padding=1) if conv_final else None

    def forward(self, x, visual=False):
        visual = self.visual
        x_dct = self.dct2d(x)
        x_idct = self.idct2d(self.att(x_dct) + x_dct)
        x = torch.cat((self.conv1(x_idct) + x, x), dim=1)
        # return self.conv2(x) if self.conv2 is not None else x   
        if not visual:
            return self.conv2(x) if self.conv2 is not None else x   
        else:
            return (
                self.conv2(x) if self.conv2 is not None else x,
                x_dct,
                self.att(x_dct),
                x_idct,
            )   


class SINet_Simp_RFDCT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCT, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        # self.rf_low_sm = nn.Sequential( RF(320, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True, visual=False)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True, visual=False)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True, visual=False)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True, visual=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True, visual=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True, visual=True)) 
        self.pdc_im = PDC_IM(channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x0 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        # x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        # x01_down = self.downSample(x01)         # (BS, 320, 44, 44)
        # x01_sm_rf = self.rf_low_sm(x01_down)   # (BS, 32, 44, 44)
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels
        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
            
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)
        # camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf, x2_im_dct, x2_im_dct_se0, x2_im_dct_se, x2_im_idct = self.rf2_im(x2_sa)
        x3_im_rf, x3_im_dct, x3_im_dct_se0, x3_im_dct_se, x3_im_idct = self.rf3_im(x3_im)
        x4_im_rf, x4_im_dct, x4_im_dct_se0, x4_im_dct_se, x4_im_idct = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return (
            # self.upsample_8(camouflage_map_sm), 
            # self.upsample_8(camouflage_map_im),
            self.upsample_8(torch.mean(x2_sa,dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(self.rf2_im[0](x2_sa),dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(x2_im_dct, dim=1).unsqueeze(dim=1)), 
            self.upsample_8(torch.mean(x2_im_dct_se0, dim=1).unsqueeze(dim=1)), 
            self.upsample_8(torch.mean(x2_im_dct_se, dim=1).unsqueeze(dim=1)), 
            self.upsample_8(torch.mean(x2_im_idct, dim=1).unsqueeze(dim=1)), 
            self.upsample_8(torch.mean(x2_im_rf, dim=1).unsqueeze(dim=1)), 
        ) 

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')


class SINet_Simp_RFDCTATT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCTATT, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCTATT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCTATT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCTATT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCTATT_Conv(channel, 8, 44, conv_final=True, visual=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCTATT_Conv(channel, 8, 22, conv_final=True, visual=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCTATT_Conv(channel, 8, 11, conv_final=True, visual=True)) 
        self.pdc_im = PDC_IM(channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x0 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x0)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        # x01 = torch.cat((x0, x1), dim=1)        # (BS, 64+256, 88, 88)
        # x01_down = self.downSample(x01)         # (BS, 320, 44, 44)
        # x01_sm_rf = self.rf_low_sm(x01_down)   # (BS, 32, 44, 44)
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels
        
        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf, x2_im_dct, x2_im_dct_att, x2_im_idct = self.rf2_im(x2_sa)
        x3_im_rf, x3_im_dct, x3_im_dct_att, x3_im_idct = self.rf3_im(x3_im)
        x4_im_rf, x4_im_dct, x4_im_dct_att, x4_im_idct = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        # return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)
        return (
            self.upsample_8(torch.mean(x2_sa,dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(self.rf2_im[0](x2_sa),dim=1).unsqueeze(dim=1)),
            self.upsample_8(torch.mean(x2_im_dct, dim=1).unsqueeze(dim=1)), 
            self.upsample_8(torch.mean(x2_im_dct_att, dim=1).unsqueeze(dim=1)), 
            self.upsample_8(torch.mean(x2_im_idct, dim=1).unsqueeze(dim=1)), 
            self.upsample_8(torch.mean(x2_im_rf, dim=1).unsqueeze(dim=1)), 
        )

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')


" ---------------- SINet Simp Dec Series ---------------------------------------"


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


class SINet_Simp_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)
        # self.dec0 = Dec(64, channel, 64//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()
        # torch.autograd.set_detect_anomaly(True)

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x1 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x1)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        out1 = self.dec1(x1, self.upsample_2(camouflage_map_im))
        # out0 = self.dec0(x0, self.upsample_2(out1))


        # ---- output ----
        # return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im), self.upsample_4(out1)
        # return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im), self.upsample_4(out1), self.upsample_2(out0)

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')


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


class SINet_Simp_Dec2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_Dec2, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec2(256, channel, 256//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x1 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x1)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        out1 = self.dec1(x1, self.upsample_2(camouflage_map_im))
        # out0 = self.dec0(x0, self.upsample_2(out1))


        # ---- output ----
        # return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im), self.upsample_4(out1)
        # return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im), self.upsample_4(out1), self.upsample_2(out0)

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')


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


class SINet_Simp_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        if self.training:
            self.initialize_weights()

    def forward(self, x):
        # ---- feature abstraction -----
        # - head
        x0 = self.resnet.conv1(x)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        # - low-level features
        x1 = self.resnet.maxpool(x0)    # (BS, 64, 88, 88)
        x1 = self.resnet.layer1(x1)     # (BS, 256, 88, 88)
        x2 = self.resnet.layer2(x1)     # (BS, 512, 44, 44)

        # ---- Stage-1: Search Module (SM) ----
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels

        x2_sm_rf = self.rf2_sm(x2_sm_cat)
        x3_sm_rf = self.rf3_sm(x3_sm_cat)
        x4_sm_rf = self.rf4_sm(x4_sm)
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # x1_1 = torch.cat([x1, (1 - self.upsample_2(camouflage_map_im).sigmoid()).repeat(1,self.dec1.tag_channel,1,1)], dim=1)
        # x1_2 = self.dec1.conv_cat[0](x1_1)
        # x1_3 = self.dec1.conv_cat[1](x1_2)
        # x1_4 = self.dec1.conv_cat[2](x1_3)

        out1 = self.dec1(x1, self.upsample_2(camouflage_map_im))
        # out0 = self.dec0(x0, self.upsample_2(out1))

        # ---- output ----
        return (
            self.upsample_8(camouflage_map_sm), 
            self.upsample_8(camouflage_map_im), 
            self.upsample_4(out1),
            # self.upsample_4(torch.mean(x1_1,dim=1).unsqueeze(dim=1)),
            # self.upsample_4(torch.mean(x1_2,dim=1).unsqueeze(dim=1)),
            # self.upsample_4(torch.mean(x1_3,dim=1).unsqueeze(dim=1)),
            # self.upsample_4(torch.mean(x1_4,dim=1).unsqueeze(dim=1)),
        ) 

    def initialize_weights(self):
        resnet50 = models.resnet50(pretrained=True)
        pretrained_dict = resnet50.state_dict()
        all_params = {}

        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())

        self.resnet.load_state_dict(all_params)
        print('[INFO] initialize weights from resnet50')



" ------------------------ SINet SimpSingle Series ------------------------------------------"


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


class SINet_SimpSingle_MTune(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTune, self).__init__()

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
        
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, _, x2_sm, x3_sm, x4_sm = self.resf(x)

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
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf) + camouflage_map_sm

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

