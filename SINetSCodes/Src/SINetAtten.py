import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
import torchvision.models as models
from .SINet_Parts import *

" -------------------- Original ------------------------------------------"

# ./Result/SINet_COD10K_AllCam_SINet_ResNet50/COD10K_all_cam 
#  {'Smeasure': 0.766, 'wFmeasure': 0.522, 'MAE': 0.055, 'adpEm': 0.791, 'meanEm': 0.79, 'maxEm': 0.866, 'adpFm': 0.582, 'meanFm': 0.615, 'maxFm': 0.669}
# ./Result/SINet_COD10K_AllCam_SINet_ResNet50/CAMO 
#  {'Smeasure': 0.678, 'wFmeasure': 0.476, 'MAE': 0.127, 'adpEm': 0.792, 'meanEm': 0.664, 'maxEm': 0.775, 'adpFm': 0.641, 'meanFm': 0.545, 'maxFm': 0.61}
# ./Result/SINet_COD10K_AllCam_SINet_ResNet50/CHAMELEON 
#  {'Smeasure': 0.848, 'wFmeasure': 0.691, 'MAE': 0.048, 'adpEm': 0.896, 'meanEm': 0.854, 'maxEm': 0.916, 'adpFm': 0.756, 'meanFm': 0.747, 'maxFm': 0.799}
# ./Result/SINet_COD10K_AllCam_SINet_ResNet50/COD10K 
#  {'Smeasure': 0.817, 'wFmeasure': 0.265, 'MAE': 0.092, 'adpEm': 0.827, 'meanEm': 0.828, 'maxEm': 0.869, 'adpFm': 0.295, 'meanFm': 0.311, 'maxFm': 0.339}
# test, train0, epoch30
# [INFO] => [2021-04-15 00:00:52] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_ResNet50/COD10K_all_cam]
# {'Smeasure': 0.766, 'wFmeasure': 0.521, 'MAE': 0.055, 'adpEm': 0.792, 'meanEm': 0.789, 'maxEm': 0.868, 'adpFm': 0.583, 'meanFm': 0.615, 'maxFm': 0.67}
# [INFO] => [2021-04-15 00:01:21] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_ResNet50/CAMO]
# {'Smeasure': 0.678, 'wFmeasure': 0.475, 'MAE': 0.128, 'adpEm': 0.791, 'meanEm': 0.664, 'maxEm': 0.772, 'adpFm': 0.64, 'meanFm': 0.544, 'maxFm': 0.608}
# [INFO] => [2021-04-15 00:01:34] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_ResNet50/CHAMELEON]
# {'Smeasure': 0.849, 'wFmeasure': 0.693, 'MAE': 0.048, 'adpEm': 0.899, 'meanEm': 0.855, 'maxEm': 0.92, 'adpFm': 0.759, 'meanFm': 0.749, 'maxFm': 0.802}
## tested, train1
#  {'Smeasure': 0.765, 'wFmeasure': 0.586, 'MAE': 0.048, 'adpEm': 0.834, 'meanEm': 0.822, 'maxEm': 0.864, 'adpFm': 0.621, 'meanFm': 0.643, 'maxFm': 0.67}
# ./Result/SINet_COD10K_AllCam_SINet_ResNet50/CAMO 
#  {'Smeasure': 0.668, 'wFmeasure': 0.5, 'MAE': 0.121, 'adpEm': 0.764, 'meanEm': 0.685, 'maxEm': 0.765, 'adpFm': 0.613, 'meanFm': 0.564, 'maxFm': 0.592}
# ./Result/SINet_COD10K_AllCam_SINet_ResNet50/CHAMELEON 
#  {'Smeasure': 0.844, 'wFmeasure': 0.739, 'MAE': 0.047, 'adpEm': 0.904, 'meanEm': 0.877, 'maxEm': 0.91, 'adpFm': 0.774, 'meanFm': 0.767, 'maxFm': 0.796}#  {'Smeasure': 0.854, 'wFmeasure': 0.753, 'MAE': 0.043, 'adpEm': 0.909, 'meanEm': 0.885, 'maxEm': 0.918, 'adpFm': 0.781, 'meanFm': 0.785, 'maxFm': 0.815}
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
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)

        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-17 01:05:04] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp/COD10K_all_cam]
# {'Smeasure': 0.768, 'wFmeasure': 0.599, 'MAE': 0.045, 'adpEm': 0.844, 'meanEm': 0.823, 'maxEm': 0.867, 'adpFm': 0.635, 'meanFm': 0.655, 'maxFm': 0.681}
# [INFO] => [2021-04-17 01:05:32] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp/CAMO]
# {'Smeasure': 0.66, 'wFmeasure': 0.492, 'MAE': 0.122, 'adpEm': 0.746, 'meanEm': 0.671, 'maxEm': 0.748, 'adpFm': 0.606, 'meanFm': 0.556, 'maxFm': 0.588}
# [INFO] => [2021-04-17 01:05:45] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp/CHAMELEON]
# {'Smeasure': 0.864, 'wFmeasure': 0.774, 'MAE': 0.039, 'adpEm': 0.918, 'meanEm': 0.902, 'maxEm': 0.93, 'adpFm': 0.796, 'meanFm': 0.801, 'maxFm': 0.828}
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

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


" -------------------- SINet PDC Series ------------------------------------------"


# [INFO] => [2021-04-22 10:49:23] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCATT/COD10K_all_cam]
# {'Smeasure': 0.767, 'wFmeasure': 0.587, 'MAE': 0.047, 'adpEm': 0.832, 'meanEm': 0.82, 'maxEm': 0.868, 'adpFm': 0.623, 'meanFm': 0.646, 'maxFm': 0.675}
# [INFO] => [2021-04-22 10:49:53] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCATT/CAMO]
# {'Smeasure': 0.664, 'wFmeasure': 0.493, 'MAE': 0.123, 'adpEm': 0.766, 'meanEm': 0.671, 'maxEm': 0.769, 'adpFm': 0.623, 'meanFm': 0.559, 'maxFm': 0.6}
# [INFO] => [2021-04-22 10:50:06] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCATT/CHAMELEON]
# {'Smeasure': 0.861, 'wFmeasure': 0.756, 'MAE': 0.042, 'adpEm': 0.911, 'meanEm': 0.888, 'maxEm': 0.924, 'adpFm': 0.786, 'meanFm': 0.788, 'maxFm': 0.821}
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

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


## rename SINet_Simp2 -> SINet_Simp_PDCN
## tested, train1
# ./Result/SINet_COD10K_AllCam_SINet_Simp2/COD10K_all_cam 
#  {'Smeasure': 0.768, 'wFmeasure': 0.573, 'MAE': 0.048, 'adpEm': 0.838, 'meanEm': 0.817, 'maxEm': 0.864, 'adpFm': 0.628, 'meanFm': 0.648, 'maxFm': 0.677}
# ./Result/SINet_COD10K_AllCam_SINet_Simp2/CAMO 
#  {'Smeasure': 0.659, 'wFmeasure': 0.479, 'MAE': 0.126, 'adpEm': 0.757, 'meanEm': 0.668, 'maxEm': 0.753, 'adpFm': 0.599, 'meanFm': 0.545, 'maxFm': 0.58}
# ./Result/SINet_COD10K_AllCam_SINet_Simp2/CHAMELEON 
#  {'Smeasure': 0.86, 'wFmeasure': 0.75, 'MAE': 0.043, 'adpEm': 0.917, 'meanEm': 0.887, 'maxEm': 0.925, 'adpFm': 0.792, 'meanFm': 0.794, 'maxFm': 0.825}
## change code, retest
# [INFO] => [2021-04-22 17:20:43] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCN/COD10K_all_cam]
# {'Smeasure': 0.768, 'wFmeasure': 0.574, 'MAE': 0.048, 'adpEm': 0.824, 'meanEm': 0.821, 'maxEm': 0.867, 'adpFm': 0.616, 'meanFm': 0.644, 'maxFm': 0.676}
# [INFO] => [2021-04-22 17:21:11] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCN/CAMO]
# {'Smeasure': 0.676, 'wFmeasure': 0.503, 'MAE': 0.119, 'adpEm': 0.767, 'meanEm': 0.681, 'maxEm': 0.763, 'adpFm': 0.626, 'meanFm': 0.573, 'maxFm': 0.6}
# [INFO] => [2021-04-22 17:21:24] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCN/CHAMELEON]
# {'Smeasure': 0.847, 'wFmeasure': 0.725, 'MAE': 0.047, 'adpEm': 0.896, 'meanEm': 0.865, 'maxEm': 0.905, 'adpFm': 0.769, 'meanFm': 0.766, 'maxFm': 0.8}
class SINet_Simp_PDCN(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_PDCN, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDCN(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDCN(channel)

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

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


## rename SINet_ResNet50N -> SINet_Simp_PDCNT
# [INFO] => [2021-04-13 21:44:15] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_ResNet50N/COD10K_all_cam]
# {'Smeasure': 0.761, 'wFmeasure': 0.563, 'MAE': 0.05, 'adpEm': 0.827, 'meanEm': 0.813, 'maxEm': 0.868, 'adpFm': 0.616, 'meanFm': 0.637, 'maxFm': 0.667}
# [INFO] => [2021-04-13 21:44:43] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_ResNet50N/CAMO]
# {'Smeasure': 0.675, 'wFmeasure': 0.504, 'MAE': 0.119, 'adpEm': 0.771, 'meanEm': 0.682, 'maxEm': 0.776, 'adpFm': 0.631, 'meanFm': 0.576, 'maxFm': 0.61}
# [INFO] => [2021-04-13 21:44:56] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_ResNet50N/CHAMELEON]
# {'Smeasure': 0.863, 'wFmeasure': 0.747, 'MAE': 0.041, 'adpEm': 0.915, 'meanEm': 0.894, 'maxEm': 0.939, 'adpFm': 0.785, 'meanFm': 0.786, 'maxFm': 0.82}
class SINet_Simp_PDCNT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_PDCNT, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDCNT(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDCNT(channel)

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
        # camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)
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

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-16 14:26:18] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCA/COD10K_all_cam]
# {'Smeasure': 0.769, 'wFmeasure': 0.59, 'MAE': 0.046, 'adpEm': 0.833, 'meanEm': 0.829, 'maxEm': 0.87, 'adpFm': 0.623, 'meanFm': 0.648, 'maxFm': 0.676}
# [INFO] => [2021-04-16 14:26:54] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCA/CAMO]
# {'Smeasure': 0.678, 'wFmeasure': 0.514, 'MAE': 0.118, 'adpEm': 0.769, 'meanEm': 0.693, 'maxEm': 0.767, 'adpFm': 0.633, 'meanFm': 0.583, 'maxFm': 0.615}
# [INFO] => [2021-04-16 14:27:09] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_PDCA/CHAMELEON]
# {'Smeasure': 0.849, 'wFmeasure': 0.735, 'MAE': 0.046, 'adpEm': 0.902, 'meanEm': 0.881, 'maxEm': 0.917, 'adpFm': 0.767, 'meanFm': 0.768, 'maxFm': 0.801}
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
        # camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)
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

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


" -------------------- SINet RF Series -----------------------------------------"


## rename SINet_FRE5 -> SINet_RFDCT
## test, train0
# ./Result/SINet_COD10K_AllCam_SINet_FRE5/COD10K_all_cam 
#  {'Smeasure': 0.773, 'wFmeasure': 0.558, 'MAE': 0.05, 'adpEm': 0.826, 'meanEm': 0.814, 'maxEm': 0.873, 'adpFm': 0.619, 'meanFm': 0.645, 'maxFm': 0.684}
# ./Result/SINet_COD10K_AllCam_SINet_FRE5/CAMO 
#  {'Smeasure': 0.682, 'wFmeasure': 0.5, 'MAE': 0.121, 'adpEm': 0.784, 'meanEm': 0.687, 'maxEm': 0.777, 'adpFm': 0.638, 'meanFm': 0.575, 'maxFm': 0.615}
# ./Result/SINet_COD10K_AllCam_SINet_FRE5/CHAMELEON 
#  {'Smeasure': 0.854, 'wFmeasure': 0.72, 'MAE': 0.045, 'adpEm': 0.905, 'meanEm': 0.876, 'maxEm': 0.927, 'adpFm': 0.77, 'meanFm': 0.766, 'maxFm': 0.803}
# ./Result/SINet_COD10K_AllCam_SINet_FRE5/COD10K 
#  {'Smeasure': 0.825, 'wFmeasure': 0.282, 'MAE': 0.085, 'adpEm': 0.848, 'meanEm': 0.844, 'maxEm': 0.876, 'adpFm': 0.314, 'meanFm': 0.327, 'maxFm': 0.347}
### test, train1
# ./Result/SINet_COD10K_AllCam_SINet_FRE5/COD10K_all_cam 
#  {'Smeasure': 0.771, 'wFmeasure': 0.601, 'MAE': 0.045, 'adpEm': 0.841, 'meanEm': 0.826, 'maxEm': 0.869, 'adpFm': 0.634, 'meanFm': 0.656, 'maxFm': 0.682}
# ./Result/SINet_COD10K_AllCam_SINet_FRE5/CAMO 
#  {'Smeasure': 0.671, 'wFmeasure': 0.505, 'MAE': 0.118, 'adpEm': 0.761, 'meanEm': 0.68, 'maxEm': 0.759, 'adpFm': 0.611, 'meanFm': 0.567, 'maxFm': 0.594}
# ./Result/SINet_COD10K_AllCam_SINet_FRE5/CHAMELEON 
#  {'Smeasure': 0.854, 'wFmeasure': 0.756, 'MAE': 0.043, 'adpEm': 0.909, 'meanEm': 0.884, 'maxEm': 0.916, 'adpFm': 0.782, 'meanFm': 0.784, 'maxFm': 0.812}
class SINet_RFDCT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None, visual_test=False):
        super(SINet_RFDCT, self).__init__()
        
        self.visual_test = visual_test
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = nn.Sequential( RF(320, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_SM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
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
        x01_sm_rf = self.rf_low_sm(x01_down)   # (BS, 32, 44, 44)
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
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


## rename SINet_FRE52 -> SINet_RFFFT
# ./Result/SINet_COD10K_AllCam_SINet_FRE52/COD10K_all_cam 
#  {'Smeasure': 0.772, 'wFmeasure': 0.604, 'MAE': 0.045, 'adpEm': 0.844, 'meanEm': 0.829, 'maxEm': 0.869, 'adpFm': 0.636, 'meanFm': 0.659, 'maxFm': 0.684}
# ./Result/SINet_COD10K_AllCam_SINet_FRE52/CAMO 
#  {'Smeasure': 0.68, 'wFmeasure': 0.519, 'MAE': 0.119, 'adpEm': 0.762, 'meanEm': 0.689, 'maxEm': 0.757, 'adpFm': 0.625, 'meanFm': 0.584, 'maxFm': 0.608}
# ./Result/SINet_COD10K_AllCam_SINet_FRE52/CHAMELEON 
#  {'Smeasure': 0.867, 'wFmeasure': 0.778, 'MAE': 0.039, 'adpEm': 0.92, 'meanEm': 0.905, 'maxEm': 0.937, 'adpFm': 0.802, 'meanFm': 0.806, 'maxFm': 0.834}
class SINet_RFFFT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_RFFFT, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = nn.Sequential( RF(320, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_SM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-16 00:00:49] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCTATT/COD10K_all_cam]
# {'Smeasure': 0.767, 'wFmeasure': 0.594, 'MAE': 0.046, 'adpEm': 0.841, 'meanEm': 0.823, 'maxEm': 0.866, 'adpFm': 0.633, 'meanFm': 0.65, 'maxFm': 0.675}
# [INFO] => [2021-04-16 00:01:18] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCTATT/CAMO]
# {'Smeasure': 0.664, 'wFmeasure': 0.496, 'MAE': 0.121, 'adpEm': 0.752, 'meanEm': 0.673, 'maxEm': 0.751, 'adpFm': 0.615, 'meanFm': 0.562, 'maxFm': 0.597}
# [INFO] => [2021-04-16 00:01:31] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCTATT/CHAMELEON]
# {'Smeasure': 0.863, 'wFmeasure': 0.766, 'MAE': 0.04, 'adpEm': 0.918, 'meanEm': 0.893, 'maxEm': 0.931, 'adpFm': 0.793, 'meanFm': 0.796, 'maxFm': 0.825}
class SINet_RFDCTATT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_RFDCTATT, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = nn.Sequential( RF(320, channel), DCTATT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf2_sm = nn.Sequential( RF(3584, channel), DCTATT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCTATT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCTATT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_SM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCTATT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCTATT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCTATT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-16 20:27:23] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT/COD10K_all_cam]
# {'Smeasure': 0.769, 'wFmeasure': 0.595, 'MAE': 0.046, 'adpEm': 0.839, 'meanEm': 0.823, 'maxEm': 0.867, 'adpFm': 0.632, 'meanFm': 0.653, 'maxFm': 0.679}
# [INFO] => [2021-04-16 20:27:52] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT/CAMO]
# {'Smeasure': 0.662, 'wFmeasure': 0.487, 'MAE': 0.122, 'adpEm': 0.742, 'meanEm': 0.666, 'maxEm': 0.743, 'adpFm': 0.598, 'meanFm': 0.549, 'maxFm': 0.579}
# [INFO] => [2021-04-16 20:28:05] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT/CHAMELEON]
# {'Smeasure': 0.863, 'wFmeasure': 0.765, 'MAE': 0.041, 'adpEm': 0.911, 'meanEm': 0.896, 'maxEm': 0.933, 'adpFm': 0.79, 'meanFm': 0.793, 'maxFm': 0.821}
class SINet_Simp_RFDCT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCT, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        # self.rf_low_sm = nn.Sequential( RF(320, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-16 18:44:06] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.6, 'MAE': 0.045, 'adpEm': 0.84, 'meanEm': 0.83, 'maxEm': 0.871, 'adpFm': 0.631, 'meanFm': 0.655, 'maxFm': 0.681}
# [INFO] => [2021-04-16 18:44:35] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT/CAMO]
# {'Smeasure': 0.676, 'wFmeasure': 0.51, 'MAE': 0.118, 'adpEm': 0.765, 'meanEm': 0.691, 'maxEm': 0.76, 'adpFm': 0.619, 'meanFm': 0.571, 'maxFm': 0.596}
# [INFO] => [2021-04-16 18:44:48] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT/CHAMELEON]
# {'Smeasure': 0.862, 'wFmeasure': 0.766, 'MAE': 0.038, 'adpEm': 0.915, 'meanEm': 0.896, 'maxEm': 0.927, 'adpFm': 0.792, 'meanFm': 0.793, 'maxFm': 0.819}
class SINet_Simp_RFFFT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFT, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        # self.rf_low_sm = nn.Sequential( RF(320, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-22 12:34:44] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTATT/COD10K_all_cam]
# {'Smeasure': 0.766, 'wFmeasure': 0.58, 'MAE': 0.047, 'adpEm': 0.831, 'meanEm': 0.819, 'maxEm': 0.867, 'adpFm': 0.62, 'meanFm': 0.643, 'maxFm': 0.673}
# [INFO] => [2021-04-22 12:35:12] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTATT/CAMO]
# {'Smeasure': 0.673, 'wFmeasure': 0.504, 'MAE': 0.121, 'adpEm': 0.768, 'meanEm': 0.682, 'maxEm': 0.763, 'adpFm': 0.626, 'meanFm': 0.57, 'maxFm': 0.604}
# [INFO] => [2021-04-22 12:35:25] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTATT/CHAMELEON]
# {'Smeasure': 0.855, 'wFmeasure': 0.745, 'MAE': 0.046, 'adpEm': 0.907, 'meanEm': 0.875, 'maxEm': 0.919, 'adpFm': 0.783, 'meanFm': 0.783, 'maxFm': 0.818}
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

        self.rf2_im = nn.Sequential( RF(512, channel), DCTATT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCTATT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCTATT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


## rename SINet_FREx -> SINet_Simp_RFDCT_Half
# ./Result/SINet_COD10K_AllCam_SINet_FREx/COD10K_all_cam 
#  {'Smeasure': 0.77, 'wFmeasure': 0.58, 'MAE': 0.048, 'adpEm': 0.826, 'meanEm': 0.822, 'maxEm': 0.871, 'adpFm': 0.619, 'meanFm': 0.645, 'maxFm': 0.679}
# ./Result/SINet_COD10K_AllCam_SINet_FREx/CAMO 
#  {'Smeasure': 0.667, 'wFmeasure': 0.49, 'MAE': 0.122, 'adpEm': 0.761, 'meanEm': 0.671, 'maxEm': 0.76, 'adpFm': 0.621, 'meanFm': 0.559, 'maxFm': 0.597}
# ./Result/SINet_COD10K_AllCam_SINet_FREx/CHAMELEON 
#  {'Smeasure': 0.857, 'wFmeasure': 0.74, 'MAE': 0.044, 'adpEm': 0.9, 'meanEm': 0.878, 'maxEm': 0.92, 'adpFm': 0.774, 'meanFm': 0.775, 'maxFm': 0.811}
# test again
# [INFO] => [2021-04-16 01:18:54] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCT_Half/COD10K_all_cam]
# {'Smeasure': 0.768, 'wFmeasure': 0.597, 'MAE': 0.046, 'adpEm': 0.842, 'meanEm': 0.826, 'maxEm': 0.868, 'adpFm': 0.632, 'meanFm': 0.654, 'maxFm': 0.679}
# [INFO] => [2021-04-16 01:19:22] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCT_Half/CAMO]
# {'Smeasure': 0.679, 'wFmeasure': 0.522, 'MAE': 0.118, 'adpEm': 0.765, 'meanEm': 0.694, 'maxEm': 0.76, 'adpFm': 0.633, 'meanFm': 0.588, 'maxFm': 0.618}
# [INFO] => [2021-04-16 01:19:35] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCT_Half/CHAMELEON]
# {'Smeasure': 0.868, 'wFmeasure': 0.777, 'MAE': 0.039, 'adpEm': 0.921, 'meanEm': 0.904, 'maxEm': 0.942, 'adpFm': 0.799, 'meanFm': 0.807, 'maxFm': 0.838}
class SINet_Simp_RFDCT_Half(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCT_Half, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        # self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-16 10:51:22] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Half/COD10K_all_cam]
# {'Smeasure': 0.771, 'wFmeasure': 0.598, 'MAE': 0.046, 'adpEm': 0.836, 'meanEm': 0.828, 'maxEm': 0.873, 'adpFm': 0.628, 'meanFm': 0.653, 'maxFm': 0.681}
# [INFO] => [2021-04-16 10:51:51] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Half/CAMO]
# {'Smeasure': 0.673, 'wFmeasure': 0.512, 'MAE': 0.12, 'adpEm': 0.766, 'meanEm': 0.688, 'maxEm': 0.763, 'adpFm': 0.63, 'meanFm': 0.577, 'maxFm': 0.611}
# [INFO] => [2021-04-16 10:52:04] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Half/CHAMELEON]
# {'Smeasure': 0.859, 'wFmeasure': 0.762, 'MAE': 0.04, 'adpEm': 0.916, 'meanEm': 0.892, 'maxEm': 0.924, 'adpFm': 0.785, 'meanFm': 0.789, 'maxFm': 0.815}
class SINet_Simp_RFFFT_Half(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFT_Half, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        # self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-16 15:27:12] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT_Half2/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.582, 'MAE': 0.047, 'adpEm': 0.831, 'meanEm': 0.823, 'maxEm': 0.871, 'adpFm': 0.625, 'meanFm': 0.651, 'maxFm': 0.683}
# [INFO] => [2021-04-16 15:27:40] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT_Half2/CAMO]
# {'Smeasure': 0.678, 'wFmeasure': 0.507, 'MAE': 0.12, 'adpEm': 0.769, 'meanEm': 0.688, 'maxEm': 0.763, 'adpFm': 0.629, 'meanFm': 0.577, 'maxFm': 0.607}
# [INFO] => [2021-04-16 15:27:53] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT_Half2/CHAMELEON]
# {'Smeasure': 0.861, 'wFmeasure': 0.746, 'MAE': 0.042, 'adpEm': 0.908, 'meanEm': 0.886, 'maxEm': 0.932, 'adpFm': 0.783, 'meanFm': 0.788, 'maxFm': 0.823}
class SINet_Simp_RFDCT_Half2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCT_Half2, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-16 22:07:18] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Half2/COD10K_all_cam]
# {'Smeasure': 0.768, 'wFmeasure': 0.598, 'MAE': 0.045, 'adpEm': 0.843, 'meanEm': 0.824, 'maxEm': 0.864, 'adpFm': 0.633, 'meanFm': 0.653, 'maxFm': 0.678}
# [INFO] => [2021-04-16 22:07:48] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Half2/CAMO]
# {'Smeasure': 0.667, 'wFmeasure': 0.503, 'MAE': 0.12, 'adpEm': 0.76, 'meanEm': 0.682, 'maxEm': 0.755, 'adpFm': 0.612, 'meanFm': 0.567, 'maxFm': 0.596}
# [INFO] => [2021-04-16 22:08:03] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Half2/CHAMELEON]
# {'Smeasure': 0.857, 'wFmeasure': 0.759, 'MAE': 0.042, 'adpEm': 0.915, 'meanEm': 0.893, 'maxEm': 0.926, 'adpFm': 0.783, 'meanFm': 0.788, 'maxFm': 0.815}
class SINet_Simp_RFFFT_Half2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFT_Half2, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
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
        x2_im_rf = self.rf2_im(x2_sa)
        x3_im_rf = self.rf3_im(x3_im)
        x4_im_rf = self.rf4_im(x4_im)
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-22 14:48:27] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2Half/COD10K_all_cam]
# {'Smeasure': 0.77, 'wFmeasure': 0.599, 'MAE': 0.045, 'adpEm': 0.84, 'meanEm': 0.83, 'maxEm': 0.87, 'adpFm': 0.633, 'meanFm': 0.657, 'maxFm': 0.683}
# [INFO] => [2021-04-22 14:48:57] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2Half/CAMO]
# {'Smeasure': 0.675, 'wFmeasure': 0.518, 'MAE': 0.117, 'adpEm': 0.764, 'meanEm': 0.699, 'maxEm': 0.768, 'adpFm': 0.628, 'meanFm': 0.588, 'maxFm': 0.61}
# [INFO] => [2021-04-22 14:49:10] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2Half/CHAMELEON]
# {'Smeasure': 0.859, 'wFmeasure': 0.761, 'MAE': 0.04, 'adpEm': 0.918, 'meanEm': 0.896, 'maxEm': 0.927, 'adpFm': 0.786, 'meanFm': 0.789, 'maxFm': 0.816}
class SINet_Simp_RF2Half(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2Half, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        # self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF2(512, channel, dilations=[1, 3, 5, 7, 9])
        self.rf3_im = RF2(1024, channel, dilations=[1, 3, 5, 7])
        self.rf4_im = RF2(2048, channel, dilations=[1, 3, 5])
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

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


# [INFO] => [2021-04-20 23:55:11] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF3Half/COD10K_all_cam]
# {'Smeasure': 0.77, 'wFmeasure': 0.586, 'MAE': 0.047, 'adpEm': 0.83, 'meanEm': 0.828, 'maxEm': 0.871, 'adpFm': 0.62, 'meanFm': 0.65, 'maxFm': 0.68}
# [INFO] => [2021-04-20 23:55:40] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF3Half/CAMO]
# {'Smeasure': 0.674, 'wFmeasure': 0.506, 'MAE': 0.121, 'adpEm': 0.772, 'meanEm': 0.691, 'maxEm': 0.77, 'adpFm': 0.627, 'meanFm': 0.577, 'maxFm': 0.608}
# [INFO] => [2021-04-20 23:55:53] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF3Half/CHAMELEON]
# {'Smeasure': 0.864, 'wFmeasure': 0.754, 'MAE': 0.041, 'adpEm': 0.912, 'meanEm': 0.888, 'maxEm': 0.927, 'adpFm': 0.782, 'meanFm': 0.79, 'maxFm': 0.826}
class SINet_Simp_RF3Half(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF3Half, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF3(512, channel, dilations=[1, 3, 5, 7, 9])
        self.rf3_im = RF3(1024, channel, dilations=[1, 3, 5, 7])
        self.rf4_im = RF3(2048, channel, dilations=[1, 3, 5])
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

        # ---- output ----
        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)

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


" ---------------- SINet Dec Series ---------------------------------------"


## rename SINet_SimpA -> SINet_Simp_Dec
# tested, train1
# ./Result/SINet_COD10K_AllCam_SINet_SimpA/COD10K_all_cam 
#  {'Smeasure': 0.773, 'wFmeasure': 0.606, 'MAE': 0.045, 'adpEm': 0.844, 'meanEm': 0.833, 'maxEm': 0.871, 'adpFm': 0.639, 'meanFm': 0.661, 'maxFm': 0.685}
# ./Result/SINet_COD10K_AllCam_SINet_SimpA/CAMO 
#  {'Smeasure': 0.677, 'wFmeasure': 0.52, 'MAE': 0.118, 'adpEm': 0.767, 'meanEm': 0.696, 'maxEm': 0.768, 'adpFm': 0.632, 'meanFm': 0.584, 'maxFm': 0.615}
# ./Result/SINet_COD10K_AllCam_SINet_SimpA/CHAMELEON 
#  {'Smeasure': 0.868, 'wFmeasure': 0.78, 'MAE': 0.039, 'adpEm': 0.929, 'meanEm': 0.905, 'maxEm': 0.941, 'adpFm': 0.807, 'meanFm': 0.811, 'maxFm': 0.838}
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


# [INFO] => [2021-04-24 00:18:27] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_DecS/COD10K_all_cam]
# {'Smeasure': 0.77, 'wFmeasure': 0.599, 'MAE': 0.045, 'adpEm': 0.837, 'meanEm': 0.832, 'maxEm': 0.874, 'adpFm': 0.629, 'meanFm': 0.655, 'maxFm': 0.681}
# [INFO] => [2021-04-24 00:18:57] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_DecS/CAMO]
# {'Smeasure': 0.683, 'wFmeasure': 0.531, 'MAE': 0.116, 'adpEm': 0.78, 'meanEm': 0.702, 'maxEm': 0.784, 'adpFm': 0.646, 'meanFm': 0.598, 'maxFm': 0.63}
# [INFO] => [2021-04-24 00:19:11] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_DecS/CHAMELEON]
# {'Smeasure': 0.867, 'wFmeasure': 0.772, 'MAE': 0.038, 'adpEm': 0.918, 'meanEm': 0.902, 'maxEm': 0.938, 'adpFm': 0.791, 'meanFm': 0.801, 'maxFm': 0.831}
class SINet_Simp_DecS(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_DecS, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = BasicConv2d(512, channel, kernel_size=3, padding=1)
        self.rf3_im = BasicConv2d(1024, channel, kernel_size=3, padding=1)
        self.rf4_im = BasicConv2d(2048, channel, kernel_size=3, padding=1)
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


## SINet_Simp_DecSS -> SINet_Simp_Basic_Dec
# [INFO] => [2021-04-24 01:52:43] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_DecSS/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.609, 'MAE': 0.044, 'adpEm': 0.85, 'meanEm': 0.831, 'maxEm': 0.869, 'adpFm': 0.646, 'meanFm': 0.663, 'maxFm': 0.685}
# [INFO] => [2021-04-24 01:53:12] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_DecSS/CAMO]
# {'Smeasure': 0.665, 'wFmeasure': 0.505, 'MAE': 0.121, 'adpEm': 0.761, 'meanEm': 0.679, 'maxEm': 0.774, 'adpFm': 0.62, 'meanFm': 0.569, 'maxFm': 0.605}
# [INFO] => [2021-04-24 01:53:26] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_DecSS/CHAMELEON]
# {'Smeasure': 0.867, 'wFmeasure': 0.783, 'MAE': 0.037, 'adpEm': 0.927, 'meanEm': 0.906, 'maxEm': 0.935, 'adpFm': 0.804, 'meanFm': 0.809, 'maxFm': 0.832}
class SINet_Simp_DecSS(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_DecSS, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential(
            BasicConv2d(512, 3*channel, kernel_size=3, padding=1),
            BasicConv2d(3*channel, channel, kernel_size=3, padding=1),
        ) 
        self.rf3_im = nn.Sequential(
            BasicConv2d(1024, 3*channel, kernel_size=3, padding=1),
            BasicConv2d(3*channel, channel, kernel_size=3, padding=1),
        ) 
        self.rf4_im = nn.Sequential(
            BasicConv2d(2048, 3*channel, kernel_size=3, padding=1),
            BasicConv2d(3*channel, channel, kernel_size=3, padding=1),
        )
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


## rename SINet_SimpB -> SINet_Simp_Dec2 
# ./Result/SINet_COD10K_AllCam_SINet_SimpB/COD10K_all_cam 
#  {'Smeasure': 0.771, 'wFmeasure': 0.603, 'MAE': 0.045, 'adpEm': 0.842, 'meanEm': 0.838, 'maxEm': 0.872, 'adpFm': 0.634, 'meanFm': 0.658, 'maxFm': 0.682}
# ./Result/SINet_COD10K_AllCam_SINet_SimpB/CAMO 
#  {'Smeasure': 0.676, 'wFmeasure': 0.52, 'MAE': 0.119, 'adpEm': 0.77, 'meanEm': 0.707, 'maxEm': 0.77, 'adpFm': 0.628, 'meanFm': 0.584, 'maxFm': 0.61}
# ./Result/SINet_COD10K_AllCam_SINet_SimpB/CHAMELEON 
#  {'Smeasure': 0.866, 'wFmeasure': 0.773, 'MAE': 0.039, 'adpEm': 0.915, 'meanEm': 0.898, 'maxEm': 0.927, 'adpFm': 0.792, 'meanFm': 0.803, 'maxFm': 0.83}
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


## rename SINet_SimpE -> SINet_Simp_Dec3 (Dec3 is not in Table Again)
# ./Result/SINet_COD10K_AllCam_SINet_SimpE/COD10K_all_cam 
#  {'Smeasure': 0.767, 'wFmeasure': 0.6, 'MAE': 0.045, 'adpEm': 0.848, 'meanEm': 0.827, 'maxEm': 0.865, 'adpFm': 0.641, 'meanFm': 0.656, 'maxFm': 0.678}
# ./Result/SINet_COD10K_AllCam_SINet_SimpE/CAMO 
#  {'Smeasure': 0.676, 'wFmeasure': 0.52, 'MAE': 0.117, 'adpEm': 0.767, 'meanEm': 0.693, 'maxEm': 0.777, 'adpFm': 0.631, 'meanFm': 0.586, 'maxFm': 0.616}
# ./Result/SINet_COD10K_AllCam_SINet_SimpE/CHAMELEON 
#  {'Smeasure': 0.862, 'wFmeasure': 0.776, 'MAE': 0.039, 'adpEm': 0.925, 'meanEm': 0.89, 'maxEm': 0.919, 'adpFm': 0.813, 'meanFm': 0.805, 'maxFm': 0.827}
class SINet_Simp_Dec3(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_Dec3, self).__init__()

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

        self.dec1 = Dec3(256, channel, 256//4)

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


## (Dec4 is Dec3 in Table)
# [INFO] => [2021-04-16 23:26:20] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_Dec4/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.61, 'MAE': 0.044, 'adpEm': 0.851, 'meanEm': 0.836, 'maxEm': 0.868, 'adpFm': 0.646, 'meanFm': 0.663, 'maxFm': 0.683}
# [INFO] => [2021-04-16 23:26:49] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_Dec4/CAMO]
# {'Smeasure': 0.666, 'wFmeasure': 0.506, 'MAE': 0.121, 'adpEm': 0.746, 'meanEm': 0.687, 'maxEm': 0.757, 'adpFm': 0.609, 'meanFm': 0.572, 'maxFm': 0.595}
# [INFO] => [2021-04-16 23:27:02] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_Dec4/CHAMELEON]
# {'Smeasure': 0.866, 'wFmeasure': 0.782, 'MAE': 0.037, 'adpEm': 0.921, 'meanEm': 0.902, 'maxEm': 0.928, 'adpFm': 0.801, 'meanFm': 0.81, 'maxFm': 0.835}
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


" -------------------- FREQRF + Dec -----------------------------"

## rename SINet_SimpAFRE5 -> SINet_Simp_RFDCT_Dec
# [INFO] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpAFRE5/COD10K_all_cam]
# {'Smeasure': 0.775, 'wFmeasure': 0.611, 'MAE': 0.044, 'adpEm': 0.846, 'meanEm': 0.836, 'maxEm': 0.87, 'adpFm': 0.643, 'meanFm': 0.665, 'maxFm': 0.688}
# [INFO] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpAFRE5/CAMO]
# {'Smeasure': 0.674, 'wFmeasure': 0.514, 'MAE': 0.119, 'adpEm': 0.763, 'meanEm': 0.695, 'maxEm': 0.771, 'adpFm': 0.622, 'meanFm': 0.579, 'maxFm': 0.609}
# [INFO] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpAFRE5/CHAMELEON]
# {'Smeasure': 0.86, 'wFmeasure': 0.773, 'MAE': 0.039, 'adpEm': 0.918, 'meanEm': 0.9, 'maxEm': 0.929, 'adpFm': 0.795, 'meanFm': 0.801, 'maxFm': 0.824}
class SINet_Simp_RFDCT_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCT_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# rename SINet_SimpAFRE52 -> SINet_Simp_RFFFT_Dec
# [INFO] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpAFRE52/COD10K_all_cam]
# {'Smeasure': 0.767, 'wFmeasure': 0.593, 'MAE': 0.046, 'adpEm': 0.836, 'meanEm': 0.821, 'maxEm': 0.868, 'adpFm': 0.629, 'meanFm': 0.651, 'maxFm': 0.68}
# [INFO] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpAFRE52/CAMO]
# {'Smeasure': 0.67, 'wFmeasure': 0.505, 'MAE': 0.121, 'adpEm': 0.764, 'meanEm': 0.681, 'maxEm': 0.766, 'adpFm': 0.624, 'meanFm': 0.57, 'maxFm': 0.607}
# [INFO] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpAFRE52/CHAMELEON]
# {'Smeasure': 0.857, 'wFmeasure': 0.753, 'MAE': 0.04, 'adpEm': 0.905, 'meanEm': 0.879, 'maxEm': 0.913, 'adpFm': 0.787, 'meanFm': 0.783, 'maxFm': 0.813}
class SINet_Simp_RFFFT_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFT_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 02:17:38] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf_Dec/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.6, 'MAE': 0.046, 'adpEm': 0.837, 'meanEm': 0.83, 'maxEm': 0.872, 'adpFm': 0.632, 'meanFm': 0.658, 'maxFm': 0.685}
# [INFO] => [2021-04-17 02:18:06] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf_Dec/CAMO]
# {'Smeasure': 0.677, 'wFmeasure': 0.52, 'MAE': 0.118, 'adpEm': 0.773, 'meanEm': 0.69, 'maxEm': 0.772, 'adpFm': 0.644, 'meanFm': 0.59, 'maxFm': 0.629}
# [INFO] => [2021-04-17 02:18:19] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf_Dec/CHAMELEON]
# {'Smeasure': 0.851, 'wFmeasure': 0.748, 'MAE': 0.045, 'adpEm': 0.902, 'meanEm': 0.88, 'maxEm': 0.912, 'adpFm': 0.779, 'meanFm': 0.782, 'maxFm': 0.811}
class SINet_Simp_RFDCTHalf_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCTHalf_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 03:31:57] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf_Dec/COD10K_all_cam]
# {'Smeasure': 0.769, 'wFmeasure': 0.596, 'MAE': 0.046, 'adpEm': 0.831, 'meanEm': 0.828, 'maxEm': 0.869, 'adpFm': 0.626, 'meanFm': 0.653, 'maxFm': 0.679}
# [INFO] => [2021-04-17 03:32:25] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf_Dec/CAMO]
# {'Smeasure': 0.68, 'wFmeasure': 0.521, 'MAE': 0.117, 'adpEm': 0.778, 'meanEm': 0.697, 'maxEm': 0.779, 'adpFm': 0.642, 'meanFm': 0.589, 'maxFm': 0.625}
# [INFO] => [2021-04-17 03:32:38] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf_Dec/CHAMELEON]
# {'Smeasure': 0.867, 'wFmeasure': 0.768, 'MAE': 0.038, 'adpEm': 0.916, 'meanEm': 0.895, 'maxEm': 0.928, 'adpFm': 0.793, 'meanFm': 0.797, 'maxFm': 0.826}
class SINet_Simp_RFFFTHalf_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFTHalf_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 04:44:19] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf2_Dec/COD10K_all_cam]
# {'Smeasure': 0.775, 'wFmeasure': 0.606, 'MAE': 0.045, 'adpEm': 0.841, 'meanEm': 0.833, 'maxEm': 0.87, 'adpFm': 0.637, 'meanFm': 0.662, 'maxFm': 0.686}
# [INFO] => [2021-04-17 04:44:48] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf2_Dec/CAMO]
# {'Smeasure': 0.675, 'wFmeasure': 0.518, 'MAE': 0.118, 'adpEm': 0.767, 'meanEm': 0.692, 'maxEm': 0.775, 'adpFm': 0.63, 'meanFm': 0.585, 'maxFm': 0.618}
# [INFO] => [2021-04-17 04:45:00] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf2_Dec/CHAMELEON]
# {'Smeasure': 0.871, 'wFmeasure': 0.779, 'MAE': 0.037, 'adpEm': 0.928, 'meanEm': 0.904, 'maxEm': 0.937, 'adpFm': 0.798, 'meanFm': 0.806, 'maxFm': 0.834}
class SINet_Simp_RFDCTHalf2_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCTHalf2_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 05:58:28] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf2_Dec/COD10K_all_cam]
# {'Smeasure': 0.774, 'wFmeasure': 0.612, 'MAE': 0.044, 'adpEm': 0.851, 'meanEm': 0.833, 'maxEm': 0.868, 'adpFm': 0.65, 'meanFm': 0.667, 'maxFm': 0.688}
# [INFO] => [2021-04-17 05:58:58] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf2_Dec/CAMO]
# {'Smeasure': 0.664, 'wFmeasure': 0.502, 'MAE': 0.12, 'adpEm': 0.751, 'meanEm': 0.678, 'maxEm': 0.763, 'adpFm': 0.613, 'meanFm': 0.567, 'maxFm': 0.602}
# [INFO] => [2021-04-17 05:59:12] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf2_Dec/CHAMELEON]
# {'Smeasure': 0.855, 'wFmeasure': 0.766, 'MAE': 0.042, 'adpEm': 0.921, 'meanEm': 0.894, 'maxEm': 0.932, 'adpFm': 0.796, 'meanFm': 0.796, 'maxFm': 0.819}
class SINet_Simp_RFFFTHalf2_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFTHalf2_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 07:09:39] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT_Dec4/COD10K_all_cam]
# {'Smeasure': 0.773, 'wFmeasure': 0.606, 'MAE': 0.045, 'adpEm': 0.844, 'meanEm': 0.834, 'maxEm': 0.869, 'adpFm': 0.64, 'meanFm': 0.66, 'maxFm': 0.683}
# [INFO] => [2021-04-17 07:10:09] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT_Dec4/CAMO]
# {'Smeasure': 0.674, 'wFmeasure': 0.514, 'MAE': 0.119, 'adpEm': 0.763, 'meanEm': 0.697, 'maxEm': 0.766, 'adpFm': 0.62, 'meanFm': 0.575, 'maxFm': 0.6}
# [INFO] => [2021-04-17 07:10:22] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCT_Dec4/CHAMELEON]
# {'Smeasure': 0.865, 'wFmeasure': 0.779, 'MAE': 0.039, 'adpEm': 0.926, 'meanEm': 0.9, 'maxEm': 0.932, 'adpFm': 0.807, 'meanFm': 0.809, 'maxFm': 0.834}
class SINet_Simp_RFDCT_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCT_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

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


# [INFO] => [2021-04-18 01:48:05] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Dec4/COD10K_all_cam]
# {'Smeasure': 0.771, 'wFmeasure': 0.604, 'MAE': 0.045, 'adpEm': 0.844, 'meanEm': 0.828, 'maxEm': 0.864, 'adpFm': 0.642, 'meanFm': 0.66, 'maxFm': 0.683}
# [INFO] => [2021-04-18 01:48:34] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Dec4/CAMO]
# {'Smeasure': 0.659, 'wFmeasure': 0.488, 'MAE': 0.122, 'adpEm': 0.747, 'meanEm': 0.669, 'maxEm': 0.758, 'adpFm': 0.605, 'meanFm': 0.553, 'maxFm': 0.587}
# [INFO] => [2021-04-18 01:48:48] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFT_Dec4/CHAMELEON]
# {'Smeasure': 0.862, 'wFmeasure': 0.771, 'MAE': 0.036, 'adpEm': 0.92, 'meanEm': 0.904, 'maxEm': 0.929, 'adpFm': 0.796, 'meanFm': 0.796, 'maxFm': 0.82}
class SINet_Simp_RFFFT_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFT_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

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


## rename SINet_SimpF -> SINet_Simp_RFDCTHalf_Dec4
#  {'Smeasure': 0.774, 'wFmeasure': 0.613, 'MAE': 0.043, 'adpEm': 0.853, 'meanEm': 0.837, 'maxEm': 0.871, 'adpFm': 0.649, 'meanFm': 0.668, 'maxFm': 0.689}
# ./Result/SINet_COD10K_AllCam_SINet_SimpF/CAMO 
#  {'Smeasure': 0.678, 'wFmeasure': 0.52, 'MAE': 0.116, 'adpEm': 0.752, 'meanEm': 0.698, 'maxEm': 0.758, 'adpFm': 0.621, 'meanFm': 0.582, 'maxFm': 0.604}
# ./Result/SINet_COD10K_AllCam_SINet_SimpF/CHAMELEON 
#  {'Smeasure': 0.864, 'wFmeasure': 0.774, 'MAE': 0.039, 'adpEm': 0.918, 'meanEm': 0.901, 'maxEm': 0.926, 'adpFm': 0.799, 'meanFm': 0.801, 'maxFm': 0.825}
class SINet_Simp_RFDCTHalf_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCTHalf_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
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


# rename SINet_SimpF1 -> SINet_Simp_RFFFTHalf_Dec4
# ./Result/SINet_COD10K_AllCam_SINet_SimpF1/COD10K_all_cam 
#  {'Smeasure': 0.772, 'wFmeasure': 0.602, 'MAE': 0.045, 'adpEm': 0.84, 'meanEm': 0.832, 'maxEm': 0.87, 'adpFm': 0.637, 'meanFm': 0.658, 'maxFm': 0.684}
# ./Result/SINet_COD10K_AllCam_SINet_SimpF1/CAMO 
#  {'Smeasure': 0.683, 'wFmeasure': 0.528, 'MAE': 0.116, 'adpEm': 0.779, 'meanEm': 0.704, 'maxEm': 0.782, 'adpFm': 0.646, 'meanFm': 0.594, 'maxFm': 0.631}
# ./Result/SINet_COD10K_AllCam_SINet_SimpF1/CHAMELEON 
#  {'Smeasure': 0.862, 'wFmeasure': 0.762, 'MAE': 0.04, 'adpEm': 0.915, 'meanEm': 0.889, 'maxEm': 0.921, 'adpFm': 0.792, 'meanFm': 0.791, 'maxFm': 0.821}
class SINet_Simp_RFFFTHalf_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFTHalf_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
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


# [INFO] => [2021-04-17 08:20:13] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf2_Dec4/COD10K_all_cam]
# {'Smeasure': 0.773, 'wFmeasure': 0.611, 'MAE': 0.045, 'adpEm': 0.846, 'meanEm': 0.841, 'maxEm': 0.872, 'adpFm': 0.643, 'meanFm': 0.666, 'maxFm': 0.689}
# [INFO] => [2021-04-17 08:20:43] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf2_Dec4/CAMO]
# {'Smeasure': 0.673, 'wFmeasure': 0.516, 'MAE': 0.119, 'adpEm': 0.761, 'meanEm': 0.689, 'maxEm': 0.771, 'adpFm': 0.628, 'meanFm': 0.581, 'maxFm': 0.614}
# [INFO] => [2021-04-17 08:20:56] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFDCTHalf2_Dec4/CHAMELEON]
# {'Smeasure': 0.866, 'wFmeasure': 0.777, 'MAE': 0.038, 'adpEm': 0.918, 'meanEm': 0.895, 'maxEm': 0.925, 'adpFm': 0.799, 'meanFm': 0.805, 'maxFm': 0.833}
class SINet_Simp_RFDCTHalf2_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCTHalf2_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
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


# [INFO] => [2021-04-17 09:32:32] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf2_Dec4/COD10K_all_cam]
# {'Smeasure': 0.771, 'wFmeasure': 0.601, 'MAE': 0.045, 'adpEm': 0.837, 'meanEm': 0.831, 'maxEm': 0.87, 'adpFm': 0.634, 'meanFm': 0.658, 'maxFm': 0.683}
# [INFO] => [2021-04-17 09:33:01] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf2_Dec4/CAMO]
# {'Smeasure': 0.683, 'wFmeasure': 0.534, 'MAE': 0.116, 'adpEm': 0.782, 'meanEm': 0.705, 'maxEm': 0.787, 'adpFm': 0.65, 'meanFm': 0.604, 'maxFm': 0.637}
# [INFO] => [2021-04-17 09:33:15] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RFFFTHalf2_Dec4/CHAMELEON]
# {'Smeasure': 0.862, 'wFmeasure': 0.761, 'MAE': 0.039, 'adpEm': 0.911, 'meanEm': 0.888, 'maxEm': 0.922, 'adpFm': 0.79, 'meanFm': 0.791, 'maxFm': 0.82}
class SINet_Simp_RFFFTHalf2_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFFFTHalf2_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
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



" --------------------- RF2 + FREQRF + Dec Series --------------------------------"


## rename SINet_SimpARF2 -> SINet_Simp_RF2Half_Dec
# [INFO] => [2021-04-13 17:28:23] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpARF2/COD10K_all_cam]
# {'Smeasure': 0.771, 'wFmeasure': 0.61, 'MAE': 0.044, 'adpEm': 0.854, 'meanEm': 0.83, 'maxEm': 0.865, 'adpFm': 0.65, 'meanFm': 0.664, 'maxFm': 0.683}
# [INFO] => [2021-04-13 17:28:51] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpARF2/CAMO]
# {'Smeasure': 0.659, 'wFmeasure': 0.492, 'MAE': 0.122, 'adpEm': 0.73, 'meanEm': 0.67, 'maxEm': 0.734, 'adpFm': 0.592, 'meanFm': 0.553, 'maxFm': 0.581}
# [INFO] => [2021-04-13 17:29:04] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpARF2/CHAMELEON]
# {'Smeasure': 0.866, 'wFmeasure': 0.784, 'MAE': 0.037, 'adpEm': 0.925, 'meanEm': 0.898, 'maxEm': 0.927, 'adpFm': 0.813, 'meanFm': 0.812, 'maxFm': 0.832}
class SINet_Simp_RF2Half_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2Half_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF2(512, channel, dilations=[1,3,5,7,9])
        self.rf3_im = RF2(1024, channel, dilations=[1,3,5,7])
        self.rf4_im = RF2(2048, channel, dilations=[1,3,5])
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


# [INFO] => [2021-04-17 18:25:56] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2Half_Dec4/COD10K_all_cam]
# {'Smeasure': 0.775, 'wFmeasure': 0.619, 'MAE': 0.043, 'adpEm': 0.855, 'meanEm': 0.837, 'maxEm': 0.871, 'adpFm': 0.657, 'meanFm': 0.674, 'maxFm': 0.693}
# [INFO] => [2021-04-17 18:26:27] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2Half_Dec4/CAMO]
# {'Smeasure': 0.674, 'wFmeasure': 0.521, 'MAE': 0.117, 'adpEm': 0.753, 'meanEm': 0.69, 'maxEm': 0.757, 'adpFm': 0.626, 'meanFm': 0.585, 'maxFm': 0.613}
# [INFO] => [2021-04-17 18:26:41] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2Half_Dec4/CHAMELEON]
# {'Smeasure': 0.856, 'wFmeasure': 0.764, 'MAE': 0.04, 'adpEm': 0.911, 'meanEm': 0.888, 'maxEm': 0.916, 'adpFm': 0.794, 'meanFm': 0.795, 'maxFm': 0.818}
# [2021/04/17 18:26:46]  => Start Train && Test && Evaluate for SINet_Simp_RF2DCT_Dec4
class SINet_Simp_RF2Half_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2Half_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF2(512, channel, dilations=[1,3,5,7,9])
        self.rf3_im = RF2(1024, channel, dilations=[1,3,5,7])
        self.rf4_im = RF2(2048, channel, dilations=[1,3,5])
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)
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


# [INFO] => [2021-04-17 10:49:15] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCT_Dec/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.6, 'MAE': 0.046, 'adpEm': 0.838, 'meanEm': 0.827, 'maxEm': 0.87, 'adpFm': 0.634, 'meanFm': 0.655, 'maxFm': 0.681}
# [INFO] => [2021-04-17 10:49:44] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCT_Dec/CAMO]
# {'Smeasure': 0.683, 'wFmeasure': 0.529, 'MAE': 0.116, 'adpEm': 0.784, 'meanEm': 0.7, 'maxEm': 0.783, 'adpFm': 0.644, 'meanFm': 0.591, 'maxFm': 0.628}
# [INFO] => [2021-04-17 10:49:57] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCT_Dec/CHAMELEON]
# {'Smeasure': 0.859, 'wFmeasure': 0.766, 'MAE': 0.042, 'adpEm': 0.917, 'meanEm': 0.893, 'maxEm': 0.93, 'adpFm': 0.793, 'meanFm': 0.797, 'maxFm': 0.826}
class SINet_Simp_RF2DCT_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2DCT_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF2(512, channel, dilations=[1,3,5,7,9]), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF2(1024, channel, dilations=[1,3,5,7]), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF2(2048, channel, dilations=[1,3,5,7]), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 12:08:58] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFT_Dec/COD10K_all_cam]
# {'Smeasure': 0.773, 'wFmeasure': 0.606, 'MAE': 0.045, 'adpEm': 0.843, 'meanEm': 0.834, 'maxEm': 0.875, 'adpFm': 0.637, 'meanFm': 0.662, 'maxFm': 0.687}
# [INFO] => [2021-04-17 12:09:28] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFT_Dec/CAMO]
# {'Smeasure': 0.677, 'wFmeasure': 0.521, 'MAE': 0.118, 'adpEm': 0.767, 'meanEm': 0.694, 'maxEm': 0.773, 'adpFm': 0.634, 'meanFm': 0.589, 'maxFm': 0.62}
# [INFO] => [2021-04-17 12:09:41] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFT_Dec/CHAMELEON]
# {'Smeasure': 0.861, 'wFmeasure': 0.768, 'MAE': 0.039, 'adpEm': 0.915, 'meanEm': 0.889, 'maxEm': 0.921, 'adpFm': 0.795, 'meanFm': 0.796, 'maxFm': 0.824}
class SINet_Simp_RF2FFT_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2FFT_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF2(512, channel, dilations=[1,3,5,7,9]), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF2(1024, channel, dilations=[1,3,5,7]), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF2(2048, channel, dilations=[1,3,5,7]), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 13:24:07] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf_Dec/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.609, 'MAE': 0.044, 'adpEm': 0.849, 'meanEm': 0.832, 'maxEm': 0.864, 'adpFm': 0.644, 'meanFm': 0.662, 'maxFm': 0.682}
# [INFO] => [2021-04-17 13:24:35] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf_Dec/CAMO]
# {'Smeasure': 0.668, 'wFmeasure': 0.505, 'MAE': 0.119, 'adpEm': 0.747, 'meanEm': 0.681, 'maxEm': 0.753, 'adpFm': 0.605, 'meanFm': 0.568, 'maxFm': 0.589}
# [INFO] => [2021-04-17 13:24:48] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf_Dec/CHAMELEON]
# {'Smeasure': 0.862, 'wFmeasure': 0.772, 'MAE': 0.037, 'adpEm': 0.926, 'meanEm': 0.901, 'maxEm': 0.931, 'adpFm': 0.798, 'meanFm': 0.799, 'maxFm': 0.82}
class SINet_Simp_RF2DCTHalf_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2DCTHalf_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF2(512, channel, dilations=[1,3,5,7,9]), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF2(1024, channel, dilations=[1,3,5,7]), DCT_Conv(channel, 8, 22, conv_final=True))
        self.rf4_im = nn.Sequential( RF2(2048, channel, dilations=[1,3,5]), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 14:40:56] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf_Dec/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.608, 'MAE': 0.044, 'adpEm': 0.848, 'meanEm': 0.834, 'maxEm': 0.868, 'adpFm': 0.642, 'meanFm': 0.663, 'maxFm': 0.683}
# [INFO] => [2021-04-17 14:41:24] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf_Dec/CAMO]
# {'Smeasure': 0.672, 'wFmeasure': 0.51, 'MAE': 0.118, 'adpEm': 0.749, 'meanEm': 0.691, 'maxEm': 0.753, 'adpFm': 0.607, 'meanFm': 0.571, 'maxFm': 0.591}
# [INFO] => [2021-04-17 14:41:37] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf_Dec/CHAMELEON]
# {'Smeasure': 0.871, 'wFmeasure': 0.788, 'MAE': 0.037, 'adpEm': 0.932, 'meanEm': 0.911, 'maxEm': 0.938, 'adpFm': 0.81, 'meanFm': 0.815, 'maxFm': 0.839}
class SINet_Simp_RF2FFTHalf_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2FFTHalf_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF2(512, channel, dilations=[1,3,5,7,9]), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF2(1024, channel, dilations=[1,3,5,7]), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF2(2048, channel, dilations=[1,3,5,7]), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 15:56:38] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf2_Dec/COD10K_all_cam]
# {'Smeasure': 0.771, 'wFmeasure': 0.61, 'MAE': 0.044, 'adpEm': 0.85, 'meanEm': 0.831, 'maxEm': 0.869, 'adpFm': 0.649, 'meanFm': 0.666, 'maxFm': 0.686}
# [INFO] => [2021-04-17 15:57:07] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf2_Dec/CAMO]
# {'Smeasure': 0.675, 'wFmeasure': 0.52, 'MAE': 0.117, 'adpEm': 0.763, 'meanEm': 0.691, 'maxEm': 0.768, 'adpFm': 0.636, 'meanFm': 0.588, 'maxFm': 0.618}
# [INFO] => [2021-04-17 15:57:20] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf2_Dec/CHAMELEON]
# {'Smeasure': 0.867, 'wFmeasure': 0.781, 'MAE': 0.038, 'adpEm': 0.928, 'meanEm': 0.903, 'maxEm': 0.935, 'adpFm': 0.809, 'meanFm': 0.812, 'maxFm': 0.837}
class SINet_Simp_RF2DCTHalf2_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2DCTHalf2_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF2(512, channel, dilations=[1,3,5,7,9]) 
        self.rf3_im = RF2(1024, channel, dilations=[1,3,5,7])
        self.rf4_im = RF2(2048, channel, dilations=[1,3,5,7])
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 17:13:52] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf2_Dec/COD10K_all_cam]
# {'Smeasure': 0.77, 'wFmeasure': 0.605, 'MAE': 0.045, 'adpEm': 0.847, 'meanEm': 0.833, 'maxEm': 0.868, 'adpFm': 0.64, 'meanFm': 0.659, 'maxFm': 0.68}
# [INFO] => [2021-04-17 17:14:22] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf2_Dec/CAMO]
# {'Smeasure': 0.674, 'wFmeasure': 0.518, 'MAE': 0.118, 'adpEm': 0.767, 'meanEm': 0.696, 'maxEm': 0.776, 'adpFm': 0.625, 'meanFm': 0.58, 'maxFm': 0.611}
# [INFO] => [2021-04-17 17:14:37] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf2_Dec/CHAMELEON]
# {'Smeasure': 0.854, 'wFmeasure': 0.76, 'MAE': 0.042, 'adpEm': 0.912, 'meanEm': 0.888, 'maxEm': 0.916, 'adpFm': 0.787, 'meanFm': 0.788, 'maxFm': 0.81}
class SINet_Simp_RF2FFTHalf2_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2FFTHalf2_Dec, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF2(512, channel, dilations=[1,3,5,7,9]) 
        self.rf3_im = RF2(1024, channel, dilations=[1,3,5,7])
        self.rf4_im = RF2(2048, channel, dilations=[1,3,5,7])
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec(256, channel, 256//4)

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


# [INFO] => [2021-04-17 19:29:58] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCT_Dec4/COD10K_all_cam]
# {'Smeasure': 0.754, 'wFmeasure': 0.573, 'MAE': 0.047, 'adpEm': 0.839, 'meanEm': 0.799, 'maxEm': 0.854, 'adpFm': 0.623, 'meanFm': 0.628, 'maxFm': 0.655}
# [INFO] => [2021-04-17 19:30:27] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCT_Dec4/CAMO]
# {'Smeasure': 0.635, 'wFmeasure': 0.445, 'MAE': 0.128, 'adpEm': 0.731, 'meanEm': 0.626, 'maxEm': 0.731, 'adpFm': 0.575, 'meanFm': 0.506, 'maxFm': 0.555}
# [INFO] => [2021-04-17 19:30:40] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCT_Dec4/CHAMELEON]
# {'Smeasure': 0.853, 'wFmeasure': 0.746, 'MAE': 0.043, 'adpEm': 0.911, 'meanEm': 0.864, 'maxEm': 0.901, 'adpFm': 0.788, 'meanFm': 0.779, 'maxFm': 0.81}
class SINet_Simp_RF2DCT_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2DCT_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF2(512, channel, dilations=[1,3,5,7,9]), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF2(1024, channel, dilations=[1,3,5,7]), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF2(2048, channel, dilations=[1,3,5,7]), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

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


# [INFO] => [2021-04-17 20:46:43] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFT_Dec4/COD10K_all_cam]
# {'Smeasure': 0.767, 'wFmeasure': 0.598, 'MAE': 0.045, 'adpEm': 0.843, 'meanEm': 0.823, 'maxEm': 0.869, 'adpFm': 0.637, 'meanFm': 0.655, 'maxFm': 0.68}
# [INFO] => [2021-04-17 20:47:11] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFT_Dec4/CAMO]
# {'Smeasure': 0.665, 'wFmeasure': 0.499, 'MAE': 0.122, 'adpEm': 0.761, 'meanEm': 0.676, 'maxEm': 0.765, 'adpFm': 0.622, 'meanFm': 0.563, 'maxFm': 0.6}
# [INFO] => [2021-04-17 20:47:24] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFT_Dec4/CHAMELEON]
# {'Smeasure': 0.853, 'wFmeasure': 0.753, 'MAE': 0.043, 'adpEm': 0.904, 'meanEm': 0.879, 'maxEm': 0.913, 'adpFm': 0.788, 'meanFm': 0.783, 'maxFm': 0.811}
class SINet_Simp_RF2FFT_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2FFT_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF2(512, channel, dilations=[1,3,5,7,9]), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF2(1024, channel, dilations=[1,3,5,7]), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF2(2048, channel, dilations=[1,3,5,7]), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

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


## rename SINet_SimpFRF2 -> SINet_Simp_RF2DCTHalf_Dec4
# [INFO] => [2021-04-13 16:10:47] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpFRF2/COD10K_all_cam]
# {'Smeasure': 0.772, 'wFmeasure': 0.609, 'MAE': 0.044, 'adpEm': 0.847, 'meanEm': 0.833, 'maxEm': 0.866, 'adpFm': 0.644, 'meanFm': 0.663, 'maxFm': 0.685}
# [INFO] => [2021-04-13 16:11:16] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpFRF2/CAMO]
# {'Smeasure': 0.668, 'wFmeasure': 0.507, 'MAE': 0.119, 'adpEm': 0.759, 'meanEm': 0.686, 'maxEm': 0.762, 'adpFm': 0.619, 'meanFm': 0.572, 'maxFm': 0.603}
# [INFO] => [2021-04-13 16:11:29] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpFRF2/CHAMELEON]
# {'Smeasure': 0.863, 'wFmeasure': 0.769, 'MAE': 0.039, 'adpEm': 0.915, 'meanEm': 0.897, 'maxEm': 0.922, 'adpFm': 0.793, 'meanFm': 0.798, 'maxFm': 0.825}
class SINet_Simp_RF2DCTHalf_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2DCTHalf_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF2(512, channel, dilations=[1,3,5,7,9]), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF2(1024, channel, dilations=[1,3,5,7]), DCT_Conv(channel, 8, 22, conv_final=True))
        self.rf4_im = nn.Sequential( RF2(2048, channel, dilations=[1,3,5]), DCT_Conv(channel, 8, 11, conv_final=True)) 
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


# [INFO] => [2021-04-17 22:00:58] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf_Dec4/COD10K_all_cam]
# {'Smeasure': 0.775, 'wFmeasure': 0.614, 'MAE': 0.043, 'adpEm': 0.851, 'meanEm': 0.84, 'maxEm': 0.871, 'adpFm': 0.648, 'meanFm': 0.667, 'maxFm': 0.688}
# [INFO] => [2021-04-17 22:01:27] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf_Dec4/CAMO]
# {'Smeasure': 0.675, 'wFmeasure': 0.52, 'MAE': 0.117, 'adpEm': 0.762, 'meanEm': 0.699, 'maxEm': 0.769, 'adpFm': 0.625, 'meanFm': 0.583, 'maxFm': 0.608}
# [INFO] => [2021-04-17 22:01:41] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf_Dec4/CHAMELEON]
# {'Smeasure': 0.867, 'wFmeasure': 0.782, 'MAE': 0.037, 'adpEm': 0.922, 'meanEm': 0.907, 'maxEm': 0.93, 'adpFm': 0.801, 'meanFm': 0.808, 'maxFm': 0.832}
# [2021/04/17 22:01:46]  => Start Train && Test && Evaluate for SINet_Simp_RF2DCTHalf2_Dec4
class SINet_Simp_RF2FFTHalf_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2FFTHalf_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential( RF2(512, channel, dilations=[1,3,5,7,9]), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF2(1024, channel, dilations=[1,3,5,7]), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF2(2048, channel, dilations=[1,3,5,7]), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

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


# [INFO] => [2021-04-17 23:14:09] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf2_Dec4/COD10K_all_cam]
# {'Smeasure': 0.771, 'wFmeasure': 0.611, 'MAE': 0.044, 'adpEm': 0.849, 'meanEm': 0.836, 'maxEm': 0.868, 'adpFm': 0.646, 'meanFm': 0.666, 'maxFm': 0.688}
# [INFO] => [2021-04-17 23:14:37] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf2_Dec4/CAMO]
# {'Smeasure': 0.673, 'wFmeasure': 0.517, 'MAE': 0.118, 'adpEm': 0.763, 'meanEm': 0.696, 'maxEm': 0.771, 'adpFm': 0.625, 'meanFm': 0.582, 'maxFm': 0.61}
# [INFO] => [2021-04-17 23:14:49] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2DCTHalf2_Dec4/CHAMELEON]
# {'Smeasure': 0.86, 'wFmeasure': 0.766, 'MAE': 0.04, 'adpEm': 0.914, 'meanEm': 0.897, 'maxEm': 0.923, 'adpFm': 0.789, 'meanFm': 0.792, 'maxFm': 0.818}
class SINet_Simp_RF2DCTHalf2_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2DCTHalf2_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF2(512, channel, dilations=[1,3,5,7,9]) 
        self.rf3_im = RF2(1024, channel, dilations=[1,3,5,7])
        self.rf4_im = RF2(2048, channel, dilations=[1,3,5,7])
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

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


# [INFO] => [2021-04-18 00:28:56] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf2_Dec4/COD10K_all_cam]
# {'Smeasure': 0.774, 'wFmeasure': 0.609, 'MAE': 0.045, 'adpEm': 0.844, 'meanEm': 0.834, 'maxEm': 0.872, 'adpFm': 0.642, 'meanFm': 0.665, 'maxFm': 0.689}
# [INFO] => [2021-04-18 00:29:24] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf2_Dec4/CAMO]
# {'Smeasure': 0.684, 'wFmeasure': 0.534, 'MAE': 0.116, 'adpEm': 0.78, 'meanEm': 0.707, 'maxEm': 0.782, 'adpFm': 0.648, 'meanFm': 0.6, 'maxFm': 0.636}
# [INFO] => [2021-04-18 00:29:37] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Simp_RF2FFTHalf2_Dec4/CHAMELEON]
# {'Smeasure': 0.868, 'wFmeasure': 0.774, 'MAE': 0.039, 'adpEm': 0.922, 'meanEm': 0.902, 'maxEm': 0.934, 'adpFm': 0.795, 'meanFm': 0.802, 'maxFm': 0.828}
class SINet_Simp_RF2FFTHalf2_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RF2FFTHalf2_Dec4, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), FFT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), FFT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), FFT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = RF2(512, channel, dilations=[1,3,5,7,9]) 
        self.rf3_im = RF2(1024, channel, dilations=[1,3,5,7])
        self.rf4_im = RF2(2048, channel, dilations=[1,3,5,7])
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

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


" --------------------- Single Branch Series ---------------------------------"


## rename SINet_Red -> SINet_MinSingle_Fine
# [INFO] => [2021-04-13 11:03:23] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red/COD10K_all_cam]
# {'Smeasure': 0.755, 'wFmeasure': 0.534, 'MAE': 0.052, 'adpEm': 0.828, 'meanEm': 0.8, 'maxEm': 0.861, 'adpFm': 0.604, 'meanFm': 0.621, 'maxFm': 0.654}
# [INFO] => [2021-04-13 11:03:51] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red/CAMO]
# {'Smeasure': 0.651, 'wFmeasure': 0.459, 'MAE': 0.128, 'adpEm': 0.756, 'meanEm': 0.653, 'maxEm': 0.753, 'adpFm': 0.6, 'meanFm': 0.53, 'maxFm': 0.585}
# [INFO] => [2021-04-13 11:04:04] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red/CHAMELEON]
# {'Smeasure': 0.85, 'wFmeasure': 0.719, 'MAE': 0.047, 'adpEm': 0.904, 'meanEm': 0.869, 'maxEm': 0.912, 'adpFm': 0.77, 'meanFm': 0.766, 'maxFm': 0.798}
class SINet_MinSingle_Fine(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_MinSingle_Fine, self).__init__()

        self.resf = Res_Features()

        self.rf2 = RF(512, channel)
        self.rf3 = RF(1024, channel)
        self.rf4 = RF(2048, channel)

        self.pdc = PDC_IM(channel)

        self.fine2 = FineTune(channel, channel, scale_factor=1)
        self.fine3 = FineTune(channel, channel, scale_factor=2)
        self.fine4 = FineTune(channel, channel, scale_factor=4)

    def forward(self, x):
        _, _, _, x2, x3, x4 = self.resf(x)

        x2 = self.rf2(x2) # /8
        x3 = self.rf3(x3) # /16
        x4 = self.rf4(x4) # /32

        out = self.pdc(x4, x3, x2) # out is same as x2
        out0 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine4(x4, out)  
        out1 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine3(x3, out)
        out2 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine2(x2, out)
        out3 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)

        return out0, out1, out2, out3


## rename SINet_Red1 -> SINet_MinSingle_Fine1
# [INFO] => [2021-04-13 14:03:52] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red1/COD10K_all_cam]
# {'Smeasure': 0.752, 'wFmeasure': 0.525, 'MAE': 0.052, 'adpEm': 0.818, 'meanEm': 0.795, 'maxEm': 0.863, 'adpFm': 0.596, 'meanFm': 0.62, 'maxFm': 0.655}
# [INFO] => [2021-04-13 14:04:22] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red1/CAMO]
# {'Smeasure': 0.656, 'wFmeasure': 0.463, 'MAE': 0.127, 'adpEm': 0.758, 'meanEm': 0.655, 'maxEm': 0.752, 'adpFm': 0.591, 'meanFm': 0.537, 'maxFm': 0.581}
# [INFO] => [2021-04-13 14:04:37] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red1/CHAMELEON]
# {'Smeasure': 0.845, 'wFmeasure': 0.71, 'MAE': 0.05, 'adpEm': 0.9, 'meanEm': 0.862, 'maxEm': 0.916, 'adpFm': 0.772, 'meanFm': 0.769, 'maxFm': 0.808}
class SINet_MinSingle_Fine1(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_MinSingle_Fine1, self).__init__()

        self.resf = Res_Features()

        self.rf2 = RF(512, channel)
        self.rf3 = RF(1024, channel)
        self.rf4 = RF(2048, channel)

        self.pdc = PDC_IM(channel)

        self.fine2 = FineTune1(channel, channel, scale_factor=1)
        self.fine3 = FineTune1(channel, channel, scale_factor=2)
        self.fine4 = FineTune1(channel, channel, scale_factor=4)

    def forward(self, x):
        _, _, _, x2, x3, x4 = self.resf(x)

        x2 = self.rf2(x2) # /8
        x3 = self.rf3(x3) # /16
        x4 = self.rf4(x4) # /32

        out = self.pdc(x4, x3, x2) # out is same as x2
        out0 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine4(x4, out)  
        out1 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine3(x3, out)
        out2 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine2(x2, out)
        out3 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)

        return out0, out1, out2, out3


## rename SINet_Red2 -> SINet_MinSingle_Fine2
# [INFO] => [2021-04-13 11:45:51] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red2/COD10K_all_cam]
# {'Smeasure': 0.749, 'wFmeasure': 0.534, 'MAE': 0.052, 'adpEm': 0.829, 'meanEm': 0.806, 'maxEm': 0.85, 'adpFm': 0.599, 'meanFm': 0.618, 'maxFm': 0.644}
# [INFO] => [2021-04-13 11:46:19] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red2/CAMO]
# {'Smeasure': 0.651, 'wFmeasure': 0.461, 'MAE': 0.13, 'adpEm': 0.738, 'meanEm': 0.657, 'maxEm': 0.731, 'adpFm': 0.571, 'meanFm': 0.531, 'maxFm': 0.559}
# [INFO] => [2021-04-13 11:46:33] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_Red2/CHAMELEON]
# {'Smeasure': 0.846, 'wFmeasure': 0.715, 'MAE': 0.048, 'adpEm': 0.902, 'meanEm': 0.877, 'maxEm': 0.919, 'adpFm': 0.761, 'meanFm': 0.767, 'maxFm': 0.8}
class SINet_MinSingle_Fine2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_MinSingle_Fine2, self).__init__()

        self.resf = Res_Features()

        self.rf2 = RF(512, channel)
        self.rf3 = RF(1024, channel)
        self.rf4 = RF(2048, channel)

        self.pdc = PDC_IM(channel)

        self.fine2 = FineTune2(channel, channel, tag_channel=channel//4, scale_factor=1)
        self.fine3 = FineTune2(channel, channel, tag_channel=channel//4, scale_factor=2)
        self.fine4 = FineTune2(channel, channel, tag_channel=channel//4, scale_factor=4)

    def forward(self, x):
        _, _, _, x2, x3, x4 = self.resf(x)

        x2 = self.rf2(x2) # /8
        x3 = self.rf3(x3) # /16
        x4 = self.rf4(x4) # /32

        out = self.pdc(x4, x3, x2) # out is same as x2
        out0 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine4(x4, out.sigmoid())  
        out1 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine3(x3, out.sigmoid())
        out2 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)

        return out0, out1, out2


## rename SINet_Red4 -> SINet_MinSingle_Fine3
## tested, train1
# ./Result/SINet_COD10K_AllCam_SINet_Red4/COD10K_all_cam 
#  {'Smeasure': 0.743, 'wFmeasure': 0.489, 'MAE': 0.059, 'adpEm': 0.835, 'meanEm': 0.788, 'maxEm': 0.853, 'adpFm': 0.602, 'meanFm': 0.611, 'maxFm': 0.646}
# ./Result/SINet_COD10K_AllCam_SINet_Red4/CAMO 
#  {'Smeasure': 0.653, 'wFmeasure': 0.446, 'MAE': 0.135, 'adpEm': 0.733, 'meanEm': 0.649, 'maxEm': 0.724, 'adpFm': 0.565, 'meanFm': 0.526, 'maxFm': 0.568}
# ./Result/SINet_COD10K_AllCam_SINet_Red4/CHAMELEON 
#  {'Smeasure': 0.85, 'wFmeasure': 0.691, 'MAE': 0.052, 'adpEm': 0.91, 'meanEm': 0.863, 'maxEm': 0.917, 'adpFm': 0.781, 'meanFm': 0.775, 'maxFm': 0.813}
class SINet_MinSingle_Fine3(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_MinSingle_Fine3, self).__init__()

        self.resf = Res_Features()

        self.rf2 = RF(512, channel)
        self.rf3 = RF(1024, channel)
        self.rf4 = RF(2048, channel)

        self.pdc = PDC_IM(channel)

        self.fine2 = FineTune3(channel, channel, tag_channel=channel//4, scale_factor=1)
        self.fine3 = FineTune3(channel, channel, tag_channel=channel//4, scale_factor=2)
        self.fine4 = FineTune3(channel, channel, tag_channel=channel//4, scale_factor=4)

    def forward(self, x):
        _, _, _, x2, x3, x4 = self.resf(x)

        x2 = self.rf2(x2) # /8
        x3 = self.rf3(x3) # /16
        x4 = self.rf4(x4) # /32

        out = self.pdc(x4, x3, x2) # out is same as x2
        out0 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine4(x4, out.sigmoid())  
        out1 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)
        out = self.fine3(x3, out.sigmoid())
        out2 = nn.functional.interpolate(out, scale_factor=8, mode='bilinear', align_corners=True)

        return out0, out1, out2


## rename SINet_Simp4 -> SINet_SimpSingle_MTune
# tested, train1
# ./Result/SINet_COD10K_AllCam_SINet_Simp4/COD10K_all_cam 
#  {'Smeasure': 0.76, 'wFmeasure': 0.577, 'MAE': 0.047, 'adpEm': 0.83, 'meanEm': 0.818, 'maxEm': 0.861, 'adpFm': 0.614, 'meanFm': 0.634, 'maxFm': 0.661}
# ./Result/SINet_COD10K_AllCam_SINet_Simp4/CAMO 
#  {'Smeasure': 0.669, 'wFmeasure': 0.504, 'MAE': 0.119, 'adpEm': 0.762, 'meanEm': 0.683, 'maxEm': 0.767, 'adpFm': 0.627, 'meanFm': 0.572, 'maxFm': 0.612}
# ./Result/SINet_COD10K_AllCam_SINet_Simp4/CHAMELEON 
#  {'Smeasure': 0.854, 'wFmeasure': 0.749, 'MAE': 0.043, 'adpEm': 0.902, 'meanEm': 0.882, 'maxEm': 0.92, 'adpFm': 0.776, 'meanFm': 0.78, 'maxFm': 0.809}
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


# [INFO] => [2021-04-24 09:59:42] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTunex/COD10K_all_cam]
# {'Smeasure': 0.759, 'wFmeasure': 0.572, 'MAE': 0.048, 'adpEm': 0.822, 'meanEm': 0.818, 'maxEm': 0.864, 'adpFm': 0.608, 'meanFm': 0.634, 'maxFm': 0.663}
# [INFO] => [2021-04-24 10:00:12] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTunex/CAMO]
# {'Smeasure': 0.678, 'wFmeasure': 0.515, 'MAE': 0.119, 'adpEm': 0.779, 'meanEm': 0.698, 'maxEm': 0.778, 'adpFm': 0.635, 'meanFm': 0.583, 'maxFm': 0.621}
# [INFO] => [2021-04-24 10:00:26] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTunex/CHAMELEON]
# {'Smeasure': 0.855, 'wFmeasure': 0.752, 'MAE': 0.043, 'adpEm': 0.908, 'meanEm': 0.894, 'maxEm': 0.936, 'adpFm': 0.777, 'meanFm': 0.785, 'maxFm': 0.818}
class SINet_SimpSingle_MTunex(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTunex, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune(512, channel, 512//4, scale_factor=1)
        self.mt3 = MTune(1024, channel, 1024//4, scale_factor=2)
        self.mt4 = MTune(2048, channel, 2048//4, scale_factor=4)
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

        x2_im_rf = self.mt2(x2_sm, tag)
        x3_im_rf = self.mt3(x3_sm, tag)
        x4_im_rf = self.mt4(x4_sm, tag)
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf) + camouflage_map_sm

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)


# [INFO] => [2021-04-24 02:54:55] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune_Dec/COD10K_all_cam]
# {'Smeasure': 0.764, 'wFmeasure': 0.577, 'MAE': 0.049, 'adpEm': 0.821, 'meanEm': 0.822, 'maxEm': 0.864, 'adpFm': 0.604, 'meanFm': 0.634, 'maxFm': 0.666}
# [INFO] => [2021-04-24 02:55:23] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune_Dec/CAMO]
# {'Smeasure': 0.688, 'wFmeasure': 0.531, 'MAE': 0.117, 'adpEm': 0.79, 'meanEm': 0.716, 'maxEm': 0.786, 'adpFm': 0.641, 'meanFm': 0.597, 'maxFm': 0.631}
# [INFO] => [2021-04-24 02:55:36] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune_Dec/CHAMELEON]
# {'Smeasure': 0.866, 'wFmeasure': 0.764, 'MAE': 0.041, 'adpEm': 0.91, 'meanEm': 0.894, 'maxEm': 0.93, 'adpFm': 0.782, 'meanFm': 0.791, 'maxFm': 0.829}
class SINet_SimpSingle_MTune_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTune_Dec, self).__init__()

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


# [INFO] => [2021-04-24 04:47:36] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA/COD10K_all_cam]
# {'Smeasure': 0.754, 'wFmeasure': 0.559, 'MAE': 0.049, 'adpEm': 0.823, 'meanEm': 0.802, 'maxEm': 0.858, 'adpFm': 0.605, 'meanFm': 0.624, 'maxFm': 0.655}
# [INFO] => [2021-04-24 04:48:04] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA/CAMO]
# {'Smeasure': 0.657, 'wFmeasure': 0.481, 'MAE': 0.123, 'adpEm': 0.763, 'meanEm': 0.662, 'maxEm': 0.755, 'adpFm': 0.613, 'meanFm': 0.547, 'maxFm': 0.592}
# [INFO] => [2021-04-24 04:48:17] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA/CHAMELEON]
# {'Smeasure': 0.861, 'wFmeasure': 0.75, 'MAE': 0.044, 'adpEm': 0.909, 'meanEm': 0.884, 'maxEm': 0.923, 'adpFm': 0.783, 'meanFm': 0.787, 'maxFm': 0.822}
class SINet_SimpSingle_MTuneGA(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTuneGA, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.GA = GA()
        self.mt2 = MTune(channel, channel, scale_factor=1)
        self.mt3 = MTune(channel, channel, scale_factor=2)
        self.mt4 = MTune(channel, channel, scale_factor=4)
        self.pdc_im = PDC_IM(channel)

        # self.dec1 = Dec4(256, channel, 256//4)
        
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
        tag = self.GA(camouflage_map_sm.sigmoid())

        x2_im_rf = self.mt2(x2_sm_rf, tag)
        x3_im_rf = self.mt3(x3_sm_rf, tag)
        x4_im_rf = self.mt4(x4_sm_rf, tag)
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # out1 = self.dec1(x1, self.upsample_2(camouflage_map_im))

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)


# [INFO] => [2021-04-24 05:49:56] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA_Dec/COD10K_all_cam]
# {'Smeasure': 0.757, 'wFmeasure': 0.572, 'MAE': 0.049, 'adpEm': 0.826, 'meanEm': 0.815, 'maxEm': 0.863, 'adpFm': 0.609, 'meanFm': 0.632, 'maxFm': 0.66}
# [INFO] => [2021-04-24 05:50:24] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA_Dec/CAMO]
# {'Smeasure': 0.664, 'wFmeasure': 0.493, 'MAE': 0.123, 'adpEm': 0.769, 'meanEm': 0.675, 'maxEm': 0.77, 'adpFm': 0.616, 'meanFm': 0.558, 'maxFm': 0.604}
# [INFO] => [2021-04-24 05:50:37] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA_Dec/CHAMELEON]
# {'Smeasure': 0.852, 'wFmeasure': 0.75, 'MAE': 0.046, 'adpEm': 0.905, 'meanEm': 0.879, 'maxEm': 0.923, 'adpFm': 0.784, 'meanFm': 0.786, 'maxFm': 0.819}
class SINet_SimpSingle_MTuneGA_Dec(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTuneGA_Dec, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.GA = GA()
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
        tag = self.GA(camouflage_map_sm.sigmoid())

        x2_im_rf = self.mt2(x2_sm_rf, tag)
        x3_im_rf = self.mt3(x3_sm_rf, tag)
        x4_im_rf = self.mt4(x4_sm_rf, tag)
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        out1 = self.dec1(x1, self.upsample_2(camouflage_map_im))

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im), self.upsample_4(out1)



# [INFO] => [2021-04-24 09:06:08] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_Decs/COD10K_all_cam]
# {'Smeasure': 0.761, 'wFmeasure': 0.58, 'MAE': 0.048, 'adpEm': 0.827, 'meanEm': 0.818, 'maxEm': 0.863, 'adpFm': 0.612, 'meanFm': 0.639, 'maxFm': 0.666}
# [INFO] => [2021-04-24 09:06:38] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_Decs/CAMO]
# {'Smeasure': 0.674, 'wFmeasure': 0.513, 'MAE': 0.118, 'adpEm': 0.781, 'meanEm': 0.692, 'maxEm': 0.787, 'adpFm': 0.642, 'meanFm': 0.583, 'maxFm': 0.627}
# [INFO] => [2021-04-24 09:06:52] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_Decs/CHAMELEON]
# {'Smeasure': 0.864, 'wFmeasure': 0.767, 'MAE': 0.042, 'adpEm': 0.91, 'meanEm': 0.903, 'maxEm': 0.945, 'adpFm': 0.786, 'meanFm': 0.798, 'maxFm': 0.832}
class SINet_SimpSingle_Decs(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_Decs, self).__init__()

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


# [INFO] => [2021-04-24 11:42:33] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_Decs2/COD10K_all_cam]
# {'Smeasure': 0.762, 'wFmeasure': 0.586, 'MAE': 0.047, 'adpEm': 0.838, 'meanEm': 0.818, 'maxEm': 0.86, 'adpFm': 0.625, 'meanFm': 0.644, 'maxFm': 0.666}
# [INFO] => [2021-04-24 11:43:01] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_Decs2/CAMO]
# {'Smeasure': 0.662, 'wFmeasure': 0.496, 'MAE': 0.122, 'adpEm': 0.761, 'meanEm': 0.673, 'maxEm': 0.767, 'adpFm': 0.629, 'meanFm': 0.561, 'maxFm': 0.606}
# [INFO] => [2021-04-24 11:43:14] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_Decs2/CHAMELEON]
# {'Smeasure': 0.854, 'wFmeasure': 0.757, 'MAE': 0.044, 'adpEm': 0.908, 'meanEm': 0.887, 'maxEm': 0.917, 'adpFm': 0.786, 'meanFm': 0.786, 'maxFm': 0.811}
class SINet_SimpSingle_Decs2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_Decs2, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.SA = SA()

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
        x2_sa = self.SA(x3_im.sigmoid(), x2_sm)
        x2_im = self.dec2(x2_sa, x3_im)
        x1_im = self.dec1(x1_sm, self.upsample_2(x2_im))

        return (
            self.upsample_8(x3_im),
            self.upsample_8(x2_im),
            self.upsample_4(x1_im),
        ) 


## with BCELoss, ResNet
# [INFO] => [2021-04-14 13:29:09] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_v2/COD10K_all_cam]
# {'Smeasure': 0.753, 'wFmeasure': 0.556, 'MAE': 0.05, 'adpEm': 0.813, 'meanEm': 0.8, 'maxEm': 0.861, 'adpFm': 0.597, 'meanFm': 0.617, 'maxFm': 0.65}
# [INFO] => [2021-04-14 13:29:38] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_v2/CAMO]
# {'Smeasure': 0.666, 'wFmeasure': 0.494, 'MAE': 0.122, 'adpEm': 0.78, 'meanEm': 0.676, 'maxEm': 0.774, 'adpFm': 0.632, 'meanFm': 0.56, 'maxFm': 0.61}
# [INFO] => [2021-04-14 13:29:51] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_v2/CHAMELEON]
# {'Smeasure': 0.852, 'wFmeasure': 0.738, 'MAE': 0.046, 'adpEm': 0.905, 'meanEm': 0.876, 'maxEm': 0.927, 'adpFm': 0.768, 'meanFm': 0.772, 'maxFm': 0.809}
from .SINetSingle import Network__ as SINet_v2


" ------------------------- Single, Use SLoss / Res2Net ---------------------------- "

## with BCELoss, Res2Net, 64 Epoch
# [INFO] => [2021-04-24 13:06:16] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_siv2/COD10K_all_cam]
# {'Smeasure': 0.794, 'wFmeasure': 0.624, 'MAE': 0.041, 'adpEm': 0.841, 'meanEm': 0.84, 'maxEm': 0.888, 'adpFm': 0.642, 'meanFm': 0.677, 'maxFm': 0.711}
# [INFO] => [2021-04-24 13:06:45] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_siv2/CAMO]
# {'Smeasure': 0.708, 'wFmeasure': 0.564, 'MAE': 0.112, 'adpEm': 0.803, 'meanEm': 0.73, 'maxEm': 0.799, 'adpFm': 0.671, 'meanFm': 0.628, 'maxFm': 0.658}
# [INFO] => [2021-04-24 13:06:58] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_siv2/CHAMELEON]
# {'Smeasure': 0.872, 'wFmeasure': 0.768, 'MAE': 0.038, 'adpEm': 0.91, 'meanEm': 0.893, 'maxEm': 0.938, 'adpFm': 0.79, 'meanFm': 0.8, 'maxFm': 0.837}
from .SINetSingle import Network as siv2

## with BCELoss, Res2Net, 64 Epoch
# [INFO] => [2021-04-24 14:24:00] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sidecs/COD10K_all_cam]
# {'Smeasure': 0.797, 'wFmeasure': 0.639, 'MAE': 0.04, 'adpEm': 0.852, 'meanEm': 0.847, 'maxEm': 0.885, 'adpFm': 0.659, 'meanFm': 0.691, 'maxFm': 0.718}
# [INFO] => [2021-04-24 14:24:30] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sidecs/CAMO]
# {'Smeasure': 0.701, 'wFmeasure': 0.561, 'MAE': 0.111, 'adpEm': 0.796, 'meanEm': 0.725, 'maxEm': 0.797, 'adpFm': 0.667, 'meanFm': 0.626, 'maxFm': 0.653}
# [INFO] => [2021-04-24 14:24:43] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sidecs/CHAMELEON]
# {'Smeasure': 0.868, 'wFmeasure': 0.772, 'MAE': 0.04, 'adpEm': 0.913, 'meanEm': 0.893, 'maxEm': 0.929, 'adpFm': 0.792, 'meanFm': 0.803, 'maxFm': 0.836}
from .SINetSingle import Single_Decs as sidecs


## with SLoss, Res2Net, 64 Epoch
# [INFO] => [2021-04-24 15:24:14] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss_v2/COD10K_all_cam]
# {'Smeasure': 0.79, 'wFmeasure': 0.65, 'MAE': 0.04, 'adpEm': 0.875, 'meanEm': 0.879, 'maxEm': 0.89, 'adpFm': 0.673, 'meanFm': 0.692, 'maxFm': 0.711}
# [INFO] => [2021-04-24 15:24:42] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss_v2/CAMO]
# {'Smeasure': 0.702, 'wFmeasure': 0.576, 'MAE': 0.11, 'adpEm': 0.777, 'meanEm': 0.748, 'maxEm': 0.78, 'adpFm': 0.651, 'meanFm': 0.635, 'maxFm': 0.644}
# [INFO] => [2021-04-24 15:24:55] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss_v2/CHAMELEON]
# {'Smeasure': 0.875, 'wFmeasure': 0.801, 'MAE': 0.034, 'adpEm': 0.937, 'meanEm': 0.943, 'maxEm': 0.96, 'adpFm': 0.812, 'meanFm': 0.821, 'maxFm': 0.84}
from .SINetSingle import Network as sloss_v2


## with SLoss, Res2Net, 64 Epoch
# [INFO] => [2021-04-24 16:49:04] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss_decs/COD10K_all_cam]
# {'Smeasure': 0.805, 'wFmeasure': 0.679, 'MAE': 0.037, 'adpEm': 0.887, 'meanEm': 0.886, 'maxEm': 0.896, 'adpFm': 0.707, 'meanFm': 0.719, 'maxFm': 0.736}
# [INFO] => [2021-04-24 16:49:41] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss_decs/CAMO]
# {'Smeasure': 0.711, 'wFmeasure': 0.591, 'MAE': 0.107, 'adpEm': 0.782, 'meanEm': 0.759, 'maxEm': 0.793, 'adpFm': 0.667, 'meanFm': 0.65, 'maxFm': 0.662}
# [INFO] => [2021-04-24 16:49:58] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss_decs/CHAMELEON]
# {'Smeasure': 0.875, 'wFmeasure': 0.814, 'MAE': 0.034, 'adpEm': 0.939, 'meanEm': 0.928, 'maxEm': 0.943, 'adpFm': 0.835, 'meanFm': 0.835, 'maxFm': 0.849}
from .SINetSingle import Single_Decs as sloss_decs


## with SLoss, Res2Net, 100 Epoch
# [INFO] => [2021-04-24 18:06:06] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100_v2/COD10K_all_cam]
# {'Smeasure': 0.788, 'wFmeasure': 0.644, 'MAE': 0.041, 'adpEm': 0.867, 'meanEm': 0.875, 'maxEm': 0.887, 'adpFm': 0.667, 'meanFm': 0.686, 'maxFm': 0.707}
# [INFO] => [2021-04-24 18:06:37] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100_v2/CAMO]
# {'Smeasure': 0.708, 'wFmeasure': 0.587, 'MAE': 0.11, 'adpEm': 0.787, 'meanEm': 0.76, 'maxEm': 0.795, 'adpFm': 0.662, 'meanFm': 0.647, 'maxFm': 0.658}
# [INFO] => [2021-04-24 18:06:51] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100_v2/CHAMELEON]
# {'Smeasure': 0.874, 'wFmeasure': 0.804, 'MAE': 0.034, 'adpEm': 0.932, 'meanEm': 0.936, 'maxEm': 0.951, 'adpFm': 0.814, 'meanFm': 0.825, 'maxFm': 0.845}
from .SINetSingle import Network as sloss100_v2


## with SLoss, Res2Net, 100 Epoch
# [INFO] => [2021-04-24 20:03:04] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100_decs/COD10K_all_cam]
# {'Smeasure': 0.8, 'wFmeasure': 0.675, 'MAE': 0.037, 'adpEm': 0.888, 'meanEm': 0.884, 'maxEm': 0.892, 'adpFm': 0.706, 'meanFm': 0.716, 'maxFm': 0.729}
# [INFO] => [2021-04-24 20:03:33] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100_decs/CAMO]
# {'Smeasure': 0.706, 'wFmeasure': 0.585, 'MAE': 0.109, 'adpEm': 0.782, 'meanEm': 0.755, 'maxEm': 0.793, 'adpFm': 0.661, 'meanFm': 0.642, 'maxFm': 0.656}
# [INFO] => [2021-04-24 20:03:45] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100_decs/CHAMELEON]
# {'Smeasure': 0.875, 'wFmeasure': 0.814, 'MAE': 0.034, 'adpEm': 0.939, 'meanEm': 0.929, 'maxEm': 0.939, 'adpFm': 0.833, 'meanFm': 0.834, 'maxFm': 0.847}
from .SINetSingle import Single_Decs as sloss100_decs


## with BCELoss, Res2Net, 100 Epoch
# [INFO] => [2021-04-25 09:34:02] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100_v2/COD10K_all_cam]
# {'Smeasure': 0.79, 'wFmeasure': 0.626, 'MAE': 0.041, 'adpEm': 0.849, 'meanEm': 0.842, 'maxEm': 0.884, 'adpFm': 0.647, 'meanFm': 0.677, 'maxFm': 0.705}
# [INFO] => [2021-04-25 09:34:32] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100_v2/CAMO]
# {'Smeasure': 0.686, 'wFmeasure': 0.531, 'MAE': 0.116, 'adpEm': 0.768, 'meanEm': 0.699, 'maxEm': 0.767, 'adpFm': 0.642, 'meanFm': 0.591, 'maxFm': 0.617}
# [INFO] => [2021-04-25 09:34:47] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100_v2/CHAMELEON]
# {'Smeasure': 0.873, 'wFmeasure': 0.776, 'MAE': 0.036, 'adpEm': 0.92, 'meanEm': 0.906, 'maxEm': 0.943, 'adpFm': 0.792, 'meanFm': 0.804, 'maxFm': 0.834}
from .SINetSingle import Network as bce100_v2


## with BCELoss, Res2Net, 100 Epoch
# [INFO] => [2021-04-25 11:30:00] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100_decs/COD10K_all_cam]
# {'Smeasure': 0.799, 'wFmeasure': 0.643, 'MAE': 0.039, 'adpEm': 0.85, 'meanEm': 0.856, 'maxEm': 0.886, 'adpFm': 0.657, 'meanFm': 0.695, 'maxFm': 0.722}
# [INFO] => [2021-04-25 11:30:31] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100_decs/CAMO]
# {'Smeasure': 0.708, 'wFmeasure': 0.574, 'MAE': 0.108, 'adpEm': 0.797, 'meanEm': 0.741, 'maxEm': 0.796, 'adpFm': 0.677, 'meanFm': 0.639, 'maxFm': 0.664}
# [INFO] => [2021-04-25 11:30:45] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100_decs/CHAMELEON]
# {'Smeasure': 0.879, 'wFmeasure': 0.79, 'MAE': 0.035, 'adpEm': 0.923, 'meanEm': 0.912, 'maxEm': 0.941, 'adpFm': 0.805, 'meanFm': 0.817, 'maxFm': 0.847}
from .SINetSingle import Single_Decs as bce100_decs


## with SLoss, Res2Net, 100 Epoch, aug
# [INFO] => [2021-04-25 07:53:29] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100aug_v2/COD10K_all_cam]
# {'Smeasure': 0.81, 'wFmeasure': 0.672, 'MAE': 0.038, 'adpEm': 0.864, 'meanEm': 0.882, 'maxEm': 0.901, 'adpFm': 0.679, 'meanFm': 0.71, 'maxFm': 0.741}
# [INFO] => [2021-04-25 07:53:58] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100aug_v2/CAMO]
# {'Smeasure': 0.73, 'wFmeasure': 0.607, 'MAE': 0.103, 'adpEm': 0.803, 'meanEm': 0.766, 'maxEm': 0.801, 'adpFm': 0.684, 'meanFm': 0.661, 'maxFm': 0.675}
# [INFO] => [2021-04-25 07:54:11] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100aug_v2/CHAMELEON]
# {'Smeasure': 0.875, 'wFmeasure': 0.791, 'MAE': 0.035, 'adpEm': 0.914, 'meanEm': 0.922, 'maxEm': 0.945, 'adpFm': 0.798, 'meanFm': 0.813, 'maxFm': 0.844}
from .SINetSingle import Network as sloss100aug_v2


## with SLoss, Res2Net, 100 Epoch, aug
# [INFO] => [2021-04-25 00:43:47] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100aug_decs/COD10K_all_cam]
# {'Smeasure': 0.822, 'wFmeasure': 0.701, 'MAE': 0.034, 'adpEm': 0.891, 'meanEm': 0.893, 'maxEm': 0.905, 'adpFm': 0.718, 'meanFm': 0.734, 'maxFm': 0.757}
# [INFO] => [2021-04-25 00:44:16] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100aug_decs/CAMO]
# {'Smeasure': 0.717, 'wFmeasure': 0.597, 'MAE': 0.106, 'adpEm': 0.787, 'meanEm': 0.759, 'maxEm': 0.786, 'adpFm': 0.672, 'meanFm': 0.652, 'maxFm': 0.661}
# [INFO] => [2021-04-25 00:44:29] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_sloss100aug_decs/CHAMELEON]
# {'Smeasure': 0.879, 'wFmeasure': 0.812, 'MAE': 0.033, 'adpEm': 0.933, 'meanEm': 0.937, 'maxEm': 0.954, 'adpFm': 0.823, 'meanFm': 0.829, 'maxFm': 0.852}
from .SINetSingle import Single_Decs as sloss100aug_decs


## with BCELoss, Res2Net, 100 Epoch, aug
# [INFO] => [2021-04-25 05:29:51] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100aug_v2/COD10K_all_cam]
# {'Smeasure': 0.813, 'wFmeasure': 0.626, 'MAE': 0.04, 'adpEm': 0.816, 'meanEm': 0.836, 'maxEm': 0.903, 'adpFm': 0.625, 'meanFm': 0.683, 'maxFm': 0.737}
# [INFO] => [2021-04-25 05:30:21] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100aug_v2/CAMO]
# {'Smeasure': 0.72, 'wFmeasure': 0.563, 'MAE': 0.111, 'adpEm': 0.817, 'meanEm': 0.725, 'maxEm': 0.799, 'adpFm': 0.682, 'meanFm': 0.62, 'maxFm': 0.661}
# [INFO] => [2021-04-25 05:30:34] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100aug_v2/CHAMELEON]
# {'Smeasure': 0.873, 'wFmeasure': 0.747, 'MAE': 0.04, 'adpEm': 0.901, 'meanEm': 0.882, 'maxEm': 0.945, 'adpFm': 0.775, 'meanFm': 0.783, 'maxFm': 0.831}
from .SINetSingle import Network as bce100aug_v2


## with BCELoss, Res2Net, 100 Epoch, aug
# [INFO] => [2021-04-25 07:30:31] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100aug_decs/COD10K_all_cam]
# {'Smeasure': 0.822, 'wFmeasure': 0.662, 'MAE': 0.036, 'adpEm': 0.842, 'meanEm': 0.856, 'maxEm': 0.903, 'adpFm': 0.659, 'meanFm': 0.71, 'maxFm': 0.752}
# [INFO] => [2021-04-25 07:31:01] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100aug_decs/CAMO]
# {'Smeasure': 0.718, 'wFmeasure': 0.575, 'MAE': 0.107, 'adpEm': 0.818, 'meanEm': 0.728, 'maxEm': 0.807, 'adpFm': 0.689, 'meanFm': 0.628, 'maxFm': 0.668}
# [INFO] => [2021-04-25 07:31:15] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_bce100aug_decs/CHAMELEON]
# {'Smeasure': 0.882, 'wFmeasure': 0.775, 'MAE': 0.036, 'adpEm': 0.909, 'meanEm': 0.894, 'maxEm': 0.95, 'adpFm': 0.789, 'meanFm': 0.805, 'maxFm': 0.851}
from .SINetSingle import Single_Decs as bce100aug_decs


# [INFO] => [2021-04-25 14:50:56] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_Single_MinDecs2/COD10K_all_cam]
# {'Smeasure': 0.822, 'wFmeasure': 0.702, 'MAE': 0.034, 'adpEm': 0.893, 'meanEm': 0.896, 'maxEm': 0.908, 'adpFm': 0.719, 'meanFm': 0.736, 'maxFm': 0.759}
# [INFO] => [2021-04-25 14:51:24] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_Single_MinDecs2/CAMO]
# {'Smeasure': 0.722, 'wFmeasure': 0.608, 'MAE': 0.106, 'adpEm': 0.8, 'meanEm': 0.768, 'maxEm': 0.804, 'adpFm': 0.686, 'meanFm': 0.663, 'maxFm': 0.676}
# [INFO] => [2021-04-25 14:51:37] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_Single_MinDecs2/CHAMELEON]
# {'Smeasure': 0.88, 'wFmeasure': 0.809, 'MAE': 0.03, 'adpEm': 0.943, 'meanEm': 0.934, 'maxEm': 0.947, 'adpFm': 0.822, 'meanFm': 0.827, 'maxFm': 0.847}
from .SINetSingle import Single_MinDecs2


# inf ------------------------------------------------------------

from .SINetSingle import Network as slossaug_v2_inf


from .SINetSingle import Single_Decs as slossaug_decs_inf


from .SINetSingle import Single_MinDecs2 as slossaug_mindecs2_inf

" ------------------------- END ------------------------------------------------ "


if __name__ == "__main__":
    pass
