
# FAULT, DO NOT RECORD ----------------------------------------------------------------------------

# ---------------- Fre Conv Series -------------------


class Fre_Conv(nn.Module):
    def __init__(self, input_channel=32, squeeze_channel=8, resolution=44, conv_final=True):
        super(Fre_Conv, self).__init__()
        self.squeeze_real = BasicConv2d(input_channel, squeeze_channel, 1)
        self.squeeze_imag = BasicConv2d(input_channel, squeeze_channel, 1)
        self.excitation_real = BasicConv2d(squeeze_channel, input_channel, 1)
        self.excitation_imag = BasicConv2d(squeeze_channel, input_channel, 1)

        self.conv1 = BasicConv2d(input_channel, input_channel, 3, padding=1)
        self.conv2 = BasicConv2d(2*input_channel, input_channel, 3, padding=1) if conv_final else None

    def forward(self, x):
        x_rfft = torch.fft.rfft(x.float())
        x_rfft_real, x_rfft_imag = x_rfft.real.type_as(x), x_rfft.imag.type_as(x)
        x_rfft_real_tmp = self.squeeze_real(x_rfft_real)
        x_rfft_imag_tmp = self.squeeze_imag(x_rfft_imag)
        x_rfft_real_tmp, x_rfft_imag_tmp = x_rfft_real_tmp - x_rfft_imag_tmp, x_rfft_real_tmp + x_rfft_imag_tmp
        x_rfft_real_tmp = self.excitation_real(x_rfft_real_tmp)
        x_rfft_imag_tmp = self.excitation_imag(x_rfft_imag_tmp)
        x_rfft_real_tmp, x_rfft_imag_tmp = x_rfft_real_tmp - x_rfft_imag_tmp, x_rfft_real_tmp + x_rfft_imag_tmp
        x_rfft_real, x_rfft_imag = x_rfft_real_tmp + x_rfft_real, x_rfft_imag_tmp + x_rfft_imag 
        x_irfft = torch.fft.irfft(torch.complex(x_rfft_real.float(), x_rfft_imag.float())).type_as(x)

        if x_irfft.size()[-1] != x.size()[-1]:
            x_irfft = nn.functional.pad(x_irfft, pad=[1,0,0,0], mode='constant', value=0)

        # print(x.size(), x_rfft.size(), x_irfft.size())
        x_irfft = self.conv1(x_irfft) + x
        x = torch.cat([x_irfft, x], dim=1)
        return self.conv2(x) if self.conv2 is not None else x  


# tested
# ./Result/SINet_COD10K_AllCam_SINet_FRE/COD10K_all_cam 
#  {'Smeasure': 0.754, 'wFmeasure': 0.483, 'MAE': 0.062, 'adpEm': 0.777, 'meanEm': 0.767, 'maxEm': 0.854, 'adpFm': 0.558, 'meanFm': 0.585, 'maxFm': 0.655}
# ./Result/SINet_COD10K_AllCam_SINet_FRE/CAMO 
#  {'Smeasure': 0.691, 'wFmeasure': 0.474, 'MAE': 0.131, 'adpEm': 0.795, 'meanEm': 0.671, 'maxEm': 0.783, 'adpFm': 0.634, 'meanFm': 0.544, 'maxFm': 0.622}
# ./Result/SINet_COD10K_AllCam_SINet_FRE/CHAMELEON 
#  {'Smeasure': 0.825, 'wFmeasure': 0.632, 'MAE': 0.061, 'adpEm': 0.884, 'meanEm': 0.83, 'maxEm': 0.914, 'adpFm': 0.722, 'meanFm': 0.7, 'maxFm': 0.759}
# ./Result/SINet_COD10K_AllCam_SINet_FRE/COD10K 
#  {'Smeasure': 0.808, 'wFmeasure': 0.245, 'MAE': 0.099, 'adpEm': 0.814, 'meanEm': 0.813, 'maxEm': 0.862, 'adpFm': 0.282, 'meanFm': 0.296, 'maxFm': 0.332}
class SINet_FRE(nn.Module): 
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None, visual_test=False):
        super(SINet_FRE, self).__init__()
        
        self.visual_test = visual_test
        
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

        self.fre_process0_1  = Fre_Conv(channel, 8, 44)
        self.fre_process2_1 = Fre_Conv(channel, 8, 44)
        self.fre_process3_1 = Fre_Conv(channel, 8, 22)
        self.fre_process4_1 = Fre_Conv(channel, 8, 11)

        self.fre_process2_2 = Fre_Conv(channel, 8, 44)
        self.fre_process3_2 = Fre_Conv(channel, 8, 22)
        self.fre_process4_2 = Fre_Conv(channel, 8, 11)
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
        x01_sm_rf = self.fre_process0_1(self.rf_low_sm(x01_down))    # (BS, 32, 44, 44)
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels
        x2_sm_rf = self.fre_process2_1(self.rf2_sm(x2_sm_cat))
        x3_sm_rf = self.fre_process3_1(self.rf3_sm(x3_sm_cat))
        x4_sm_rf = self.fre_process4_1(self.rf4_sm(x4_sm))
            
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf = self.fre_process2_2(self.rf2_im(x2_sa))
        x3_im_rf = self.fre_process3_2(self.rf3_im(x3_im))
        x4_im_rf = self.fre_process4_2(self.rf4_im(x4_im))
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        if not self.visual_test:
            return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)
        else:
            return [self.upsample_8(camouflage_map_sm), x4_sm, x3_sm, x2, x01], [self.upsample_8(camouflage_map_im), x4_im, x3_im, x2, x01]

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


# tested
# ./Result/SINet_COD10K_AllCam_SINet_FRE2/COD10K_all_cam 
#  {'Smeasure': 0.767, 'wFmeasure': 0.541, 'MAE': 0.054, 'adpEm': 0.782, 'meanEm': 0.784, 'maxEm': 0.863, 'adpFm': 0.573, 'meanFm': 0.609, 'maxFm': 0.668}
# ./Result/SINet_COD10K_AllCam_SINet_FRE2/CAMO 
#  {'Smeasure': 0.687, 'wFmeasure': 0.499, 'MAE': 0.125, 'adpEm': 0.794, 'meanEm': 0.678, 'maxEm': 0.79, 'adpFm': 0.645, 'meanFm': 0.556, 'maxFm': 0.63}
# ./Result/SINet_COD10K_AllCam_SINet_FRE2/CHAMELEON 
#  {'Smeasure': 0.836, 'wFmeasure': 0.679, 'MAE': 0.054, 'adpEm': 0.884, 'meanEm': 0.841, 'maxEm': 0.922, 'adpFm': 0.737, 'meanFm': 0.718, 'maxFm': 0.77}
# ./Result/SINet_COD10K_AllCam_SINet_FRE2/COD10K 
#  {'Smeasure': 0.818, 'wFmeasure': 0.274, 'MAE': 0.091, 'adpEm': 0.819, 'meanEm': 0.825, 'maxEm': 0.865, 'adpFm': 0.29, 'meanFm': 0.308, 'maxFm': 0.338}
class SINet_FRE2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None, visual_test=False):
        super(SINet_FRE2, self).__init__()
        
        self.visual_test = visual_test
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_SM(2 * channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(2 * channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        self.fre_process0_1  = Fre_Conv(channel, 8, 44, conv_final=False)
        self.fre_process2_1 = Fre_Conv(channel, 8, 44, conv_final=False)
        self.fre_process3_1 = Fre_Conv(channel, 8, 22, conv_final=False)
        self.fre_process4_1 = Fre_Conv(channel, 8, 11, conv_final=False)

        self.fre_process2_2 = Fre_Conv(channel, 8, 44, conv_final=False)
        self.fre_process3_2 = Fre_Conv(channel, 8, 22, conv_final=False)
        self.fre_process4_2 = Fre_Conv(channel, 8, 11, conv_final=False)
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
        x01_sm_rf = self.fre_process0_1(self.rf_low_sm(x01_down))    # (BS, 32, 44, 44)
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels
        x2_sm_rf = self.fre_process2_1(self.rf2_sm(x2_sm_cat))
        x3_sm_rf = self.fre_process3_1(self.rf3_sm(x3_sm_cat))
        x4_sm_rf = self.fre_process4_1(self.rf4_sm(x4_sm))
            
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf = self.fre_process2_2(self.rf2_im(x2_sa))
        x3_im_rf = self.fre_process3_2(self.rf3_im(x3_im))
        x4_im_rf = self.fre_process4_2(self.rf4_im(x4_im))
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        if not self.visual_test:
            return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)
        else:
            return [self.upsample_8(camouflage_map_sm), x4_sm, x3_sm, x2, x01], [self.upsample_8(camouflage_map_im), x4_im, x3_im, x2, x01]

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


# tested
# ./Result/SINet_COD10K_AllCam_SINet_FRE3/COD10K_all_cam 
#  {'Smeasure': 0.767, 'wFmeasure': 0.538, 'MAE': 0.052, 'adpEm': 0.823, 'meanEm': 0.81, 'maxEm': 0.87, 'adpFm': 0.613, 'meanFm': 0.637, 'maxFm': 0.678}
# ./Result/SINet_COD10K_AllCam_SINet_FRE3/CAMO 
#  {'Smeasure': 0.675, 'wFmeasure': 0.486, 'MAE': 0.125, 'adpEm': 0.775, 'meanEm': 0.676, 'maxEm': 0.765, 'adpFm': 0.626, 'meanFm': 0.563, 'maxFm': 0.607}
# ./Result/SINet_COD10K_AllCam_SINet_FRE3/CHAMELEON 
#  {'Smeasure': 0.852, 'wFmeasure': 0.707, 'MAE': 0.05, 'adpEm': 0.901, 'meanEm': 0.866, 'maxEm': 0.922, 'adpFm': 0.773, 'meanFm': 0.766, 'maxFm': 0.809}
# ./Result/SINet_COD10K_AllCam_SINet_FRE3/COD10K 
#  {'Smeasure': 0.82, 'wFmeasure': 0.273, 'MAE': 0.088, 'adpEm': 0.847, 'meanEm': 0.84, 'maxEm': 0.875, 'adpFm': 0.311, 'meanFm': 0.323, 'maxFm': 0.343}
class SINet_FRE3(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None, visual_test=False):
        super(SINet_FRE3, self).__init__()
        
        self.visual_test = visual_test
        
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

        self.fre_process0_1  = Fre_Conv(channel, channel, 44)
        self.fre_process2_1 = Fre_Conv(channel, channel, 44)
        self.fre_process3_1 = Fre_Conv(channel, channel, 22)
        self.fre_process4_1 = Fre_Conv(channel, channel, 11)

        self.fre_process2_2 = Fre_Conv(channel, channel, 44)
        self.fre_process3_2 = Fre_Conv(channel, channel, 22)
        self.fre_process4_2 = Fre_Conv(channel, channel, 11)
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
        x01_sm_rf = self.fre_process0_1(self.rf_low_sm(x01_down))    # (BS, 32, 44, 44)
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels
        x2_sm_rf = self.fre_process2_1(self.rf2_sm(x2_sm_cat))
        x3_sm_rf = self.fre_process3_1(self.rf3_sm(x3_sm_cat))
        x4_sm_rf = self.fre_process4_1(self.rf4_sm(x4_sm))
            
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf = self.fre_process2_2(self.rf2_im(x2_sa))
        x3_im_rf = self.fre_process3_2(self.rf3_im(x3_im))
        x4_im_rf = self.fre_process4_2(self.rf4_im(x4_im))
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        if not self.visual_test:
            return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)
        else:
            return [self.upsample_8(camouflage_map_sm), x4_sm, x3_sm, x2, x01], [self.upsample_8(camouflage_map_im), x4_im, x3_im, x2, x01]

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

# --- No Reason ------------------

# tested
# ./Result/SINet_COD10K_AllCam_SINet_FRE4/COD10K_all_cam 
#  {'Smeasure': 0.752, 'wFmeasure': 0.504, 'MAE': 0.061, 'adpEm': 0.765, 'meanEm': 0.76, 'maxEm': 0.851, 'adpFm': 0.553, 'meanFm': 0.583, 'maxFm': 0.653}
# ./Result/SINet_COD10K_AllCam_SINet_FRE4/CAMO 
#  {'Smeasure': 0.681, 'wFmeasure': 0.477, 'MAE': 0.134, 'adpEm': 0.788, 'meanEm': 0.665, 'maxEm': 0.785, 'adpFm': 0.628, 'meanFm': 0.535, 'maxFm': 0.619}
# ./Result/SINet_COD10K_AllCam_SINet_FRE4/CHAMELEON 
#  {'Smeasure': 0.827, 'wFmeasure': 0.644, 'MAE': 0.066, 'adpEm': 0.865, 'meanEm': 0.82, 'maxEm': 0.908, 'adpFm': 0.713, 'meanFm': 0.697, 'maxFm': 0.77}
# ./Result/SINet_COD10K_AllCam_SINet_FRE4/COD10K 
#  {'Smeasure': 0.809, 'wFmeasure': 0.255, 'MAE': 0.096, 'adpEm': 0.806, 'meanEm': 0.812, 'maxEm': 0.86, 'adpFm': 0.28, 'meanFm': 0.295, 'maxFm': 0.33}
########### decay epoch 28
# ./Result/SINet_COD10K_AllCam_SINet_FRE4/COD10K_all_cam 
#  {'Smeasure': 0.77, 'wFmeasure': 0.576, 'MAE': 0.048, 'adpEm': 0.82, 'meanEm': 0.811, 'maxEm': 0.869, 'adpFm': 0.609, 'meanFm': 0.638, 'maxFm': 0.676}
# ./Result/SINet_COD10K_AllCam_SINet_FRE4/CAMO 
#  {'Smeasure': 0.666, 'wFmeasure': 0.488, 'MAE': 0.122, 'adpEm': 0.779, 'meanEm': 0.663, 'maxEm': 0.768, 'adpFm': 0.633, 'meanFm': 0.55, 'maxFm': 0.603}
# ./Result/SINet_COD10K_AllCam_SINet_FRE4/CHAMELEON 
#  {'Smeasure': 0.852, 'wFmeasure': 0.728, 'MAE': 0.047, 'adpEm': 0.895, 'meanEm': 0.863, 'maxEm': 0.911, 'adpFm': 0.766, 'meanFm': 0.766, 'maxFm': 0.807}
# ./Result/SINet_COD10K_AllCam_SINet_FRE4/COD10K 
#  {'Smeasure': 0.825, 'wFmeasure': 0.292, 'MAE': 0.083, 'adpEm': 0.846, 'meanEm': 0.844, 'maxEm': 0.876, 'adpFm': 0.309, 'meanFm': 0.323, 'maxFm': 0.343}
class SINet_FRE4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None, visual_test=False):
        super(SINet_FRE4, self).__init__()
        
        self.visual_test = visual_test
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_SM(2*channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC_IM(2*channel)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

        self.fre_process0_1  = DCT_Conv(channel, 8, 44)
        self.fre_process2_1 = DCT_Conv(channel, 8, 44)
        self.fre_process3_1 = DCT_Conv(channel, 8, 22)
        self.fre_process4_1 = DCT_Conv(channel, 8, 11)

        self.fre_process2_2 = DCT_Conv(channel, 8, 44)
        self.fre_process3_2 = DCT_Conv(channel, 8, 22)
        self.fre_process4_2 = DCT_Conv(channel, 8, 11)
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
        x01_sm_rf = self.fre_process0_1(self.rf_low_sm(x01_down))    # (BS, 32, 44, 44)
        x2_sm = x2                              # (512, 44, 44)
        x3_sm = self.resnet.layer3_1(x2_sm)     # (1024, 22, 22)
        x4_sm = self.resnet.layer4_1(x3_sm)     # (2048, 11, 11)

        x2_sm_cat = torch.cat((x2_sm,
                               self.upsample_2(x3_sm),
                               self.upsample_2(self.upsample_2(x4_sm))), dim=1)   # 3584 channels
        x3_sm_cat = torch.cat((x3_sm,
                               self.upsample_2(x4_sm)), dim=1)                    # 3072 channels
        x2_sm_rf = self.fre_process2_1(self.rf2_sm(x2_sm_cat))
        x3_sm_rf = self.fre_process3_1(self.rf3_sm(x3_sm_cat))
        x4_sm_rf = self.fre_process4_1(self.rf4_sm(x4_sm))
            
            
        camouflage_map_sm = self.pdc_sm(x4_sm_rf, x3_sm_rf, x2_sm_rf, x01_sm_rf)

        # ---- Switcher: Search Attention (SA) ----
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2)    # (512, 44, 44)

        # ---- Stage-2: Identification Module (IM) ----
        x3_im = self.resnet.layer3_2(x2_sa)                 # (1024, 22, 22)
        x4_im = self.resnet.layer4_2(x3_im)                 # (2048, 11, 11)
        x2_im_rf = self.fre_process2_2(self.rf2_im(x2_sa))
        x3_im_rf = self.fre_process3_2(self.rf3_im(x3_im))
        x4_im_rf = self.fre_process4_2(self.rf4_im(x4_im))
        # - decoder
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        # ---- output ----
        if not self.visual_test:
            return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)
        else:
            return [self.upsample_8(camouflage_map_sm), x4_sm, x3_sm, x2, x01], [self.upsample_8(camouflage_map_im), x4_im, x3_im, x2, x01]

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

class PDC3(nn.Module):
    # Partial Decoder Component (Identification Module)
    def __init__(self, channel):
        super(PDC3, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample1 = nn.ConvTranspose2d(channel, channel, 2, 2)
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(channel, channel, 2, 2),
            nn.ConvTranspose2d(channel, channel, 2, 2),
        ) 
        self.upsample3 = nn.ConvTranspose2d(channel, channel, 2, 2)
        self.upsample4 = nn.ConvTranspose2d(channel, channel, 2, 2)
        self.upsample5 = nn.ConvTranspose2d(2*channel, 2*channel, 2, 2)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.upsample1(x1) * x2
        x3_1 = self.upsample2(x1) * self.upsample3(x2) * x3

        x2_2 = torch.cat((x2_1, self.upsample4(x1_1)), 1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.cat((x3_1, self.upsample5(x2_2)), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


# untested
class SINet_Simp_PDC3(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_PDC3, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC3(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC3(channel)

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


# if SINet_Simp_PDC3 is tested this would be deleted
## rename SINet_SimpD -> SINet_Simp_PDC3_Dec2
# ./Result/SINet_COD10K_AllCam_SINet_SimpD/COD10K_all_cam 
#  {'Smeasure': 0.769, 'wFmeasure': 0.608, 'MAE': 0.045, 'adpEm': 0.852, 'meanEm': 0.834, 'maxEm': 0.866, 'adpFm': 0.647, 'meanFm': 0.663, 'maxFm': 0.681}
# ./Result/SINet_COD10K_AllCam_SINet_SimpD/CAMO 
#  {'Smeasure': 0.662, 'wFmeasure': 0.496, 'MAE': 0.12, 'adpEm': 0.744, 'meanEm': 0.674, 'maxEm': 0.758, 'adpFm': 0.603, 'meanFm': 0.562, 'maxFm': 0.592}
# ./Result/SINet_COD10K_AllCam_SINet_SimpD/CHAMELEON 
#  {'Smeasure': 0.859, 'wFmeasure': 0.774, 'MAE': 0.04, 'adpEm': 0.913, 'meanEm': 0.887, 'maxEm': 0.912, 'adpFm': 0.805, 'meanFm': 0.804, 'maxFm': 0.824}
class SINet_Simp_PDC3_Dec2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_PDC3_Dec2, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC3(channel)

        self.rf2_im = RF(512, channel)
        self.rf3_im = RF(1024, channel)
        self.rf4_im = RF(2048, channel)
        self.pdc_im = PDC3(channel)

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



## rename SINet_SimpF3 -> SINet_Simp_RFDCTHalf_Dec4_PDCA
# if SINet_Simp_PDCA is tested, this would be deleted
# ./Result/SINet_COD10K_AllCam_SINet_SimpF3/COD10K_all_cam 
#  {'Smeasure': 0.763, 'wFmeasure': 0.586, 'MAE': 0.047, 'adpEm': 0.834, 'meanEm': 0.814, 'maxEm': 0.866, 'adpFm': 0.631, 'meanFm': 0.647, 'maxFm': 0.676}
# ./Result/SINet_COD10K_AllCam_SINet_SimpF3/CAMO 
#  {'Smeasure': 0.666, 'wFmeasure': 0.503, 'MAE': 0.12, 'adpEm': 0.769, 'meanEm': 0.673, 'maxEm': 0.775, 'adpFm': 0.631, 'meanFm': 0.569, 'maxFm': 0.61}
# ./Result/SINet_COD10K_AllCam_SINet_SimpF3/CHAMELEON 
#  {'Smeasure': 0.853, 'wFmeasure': 0.745, 'MAE': 0.043, 'adpEm': 0.907, 'meanEm': 0.875, 'maxEm': 0.913, 'adpFm': 0.778, 'meanFm': 0.777, 'maxFm': 0.808}
class SINet_Simp_RFDCTHalf_Dec4_PDCA(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCTHalf_Dec4_PDCA, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDCA(channel)

        self.rf2_im = nn.Sequential( RF(512, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_im = nn.Sequential( RF(1024, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_im = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_im = PDCA(channel)

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



## may delete ---------------
# [INFO] => [2021-04-16 01:18:54] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCT_Half/COD10K_all_cam]
# {'Smeasure': 0.768, 'wFmeasure': 0.597, 'MAE': 0.046, 'adpEm': 0.842, 'meanEm': 0.826, 'maxEm': 0.868, 'adpFm': 0.632, 'meanFm': 0.654, 'maxFm': 0.679}
# [INFO] => [2021-04-16 01:19:22] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCT_Half/CAMO]
# {'Smeasure': 0.679, 'wFmeasure': 0.522, 'MAE': 0.118, 'adpEm': 0.765, 'meanEm': 0.694, 'maxEm': 0.76, 'adpFm': 0.633, 'meanFm': 0.588, 'maxFm': 0.618}
# [INFO] => [2021-04-16 01:19:35] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFDCT_Half/CHAMELEON]
# {'Smeasure': 0.868, 'wFmeasure': 0.777, 'MAE': 0.039, 'adpEm': 0.921, 'meanEm': 0.904, 'maxEm': 0.942, 'adpFm': 0.799, 'meanFm': 0.807, 'maxFm': 0.838}
class SINet_RFDCT_Half(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_RFDCT_Half, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
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


## may delete ---------------
# [INFO] => [2021-04-16 09:43:52] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFFFT_Half/COD10K_all_cam]
# {'Smeasure': 0.764, 'wFmeasure': 0.556, 'MAE': 0.05, 'adpEm': 0.832, 'meanEm': 0.819, 'maxEm': 0.868, 'adpFm': 0.621, 'meanFm': 0.641, 'maxFm': 0.674}
# [INFO] => [2021-04-16 09:44:24] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFFFT_Half/CAMO]
# {'Smeasure': 0.682, 'wFmeasure': 0.509, 'MAE': 0.12, 'adpEm': 0.776, 'meanEm': 0.691, 'maxEm': 0.777, 'adpFm': 0.64, 'meanFm': 0.587, 'maxFm': 0.621}
# [INFO] => [2021-04-16 09:44:39] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_RFFFT_Half/CHAMELEON]
# {'Smeasure': 0.867, 'wFmeasure': 0.744, 'MAE': 0.043, 'adpEm': 0.913, 'meanEm': 0.891, 'maxEm': 0.934, 'adpFm': 0.791, 'meanFm': 0.795, 'maxFm': 0.833}
class SINet_RFFFT_Half(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_RFFFT_Half, self).__init__()
        
        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
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


## rename SINet_Simp9 -> SINet_SimpSingle_SAMTune
# tested, train1
# ./Result/SINet_COD10K_AllCam_SINet_Simp9/COD10K_all_cam 
#  {'Smeasure': 0.758, 'wFmeasure': 0.559, 'MAE': 0.05, 'adpEm': 0.823, 'meanEm': 0.81, 'maxEm': 0.862, 'adpFm': 0.605, 'meanFm': 0.63, 'maxFm': 0.661}
# ./Result/SINet_COD10K_AllCam_SINet_Simp9/CAMO 
#  {'Smeasure': 0.664, 'wFmeasure': 0.491, 'MAE': 0.123, 'adpEm': 0.775, 'meanEm': 0.672, 'maxEm': 0.774, 'adpFm': 0.624, 'meanFm': 0.558, 'maxFm': 0.607}
# ./Result/SINet_COD10K_AllCam_SINet_Simp9/CHAMELEON 
#  {'Smeasure': 0.856, 'wFmeasure': 0.744, 'MAE': 0.044, 'adpEm': 0.91, 'meanEm': 0.881, 'maxEm': 0.927, 'adpFm': 0.781, 'meanFm': 0.783, 'maxFm': 0.817}
class SINet_SimpSingle_SAMTune(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_SAMTune, self).__init__()

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
        self.SA = MTune(512, 512, 512//4, 1)

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
        x2_sa = self.SA(x2, camouflage_map_sm.sigmoid())    # (512, 44, 44)

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



## rename SINet_SimpC -> SINet_Simp_RFDCT_Dec2
# ./Result/SINet_COD10K_AllCam_SINet_SimpC/COD10K_all_cam 
#  {'Smeasure': 0.77, 'wFmeasure': 0.603, 'MAE': 0.045, 'adpEm': 0.844, 'meanEm': 0.83, 'maxEm': 0.871, 'adpFm': 0.641, 'meanFm': 0.66, 'maxFm': 0.684}
# ./Result/SINet_COD10K_AllCam_SINet_SimpC/CAMO 
#  {'Smeasure': 0.673, 'wFmeasure': 0.515, 'MAE': 0.119, 'adpEm': 0.761, 'meanEm': 0.685, 'maxEm': 0.771, 'adpFm': 0.63, 'meanFm': 0.583, 'maxFm': 0.617}
# ./Result/SINet_COD10K_AllCam_SINet_SimpC/CHAMELEON 
#  {'Smeasure': 0.862, 'wFmeasure': 0.771, 'MAE': 0.04, 'adpEm': 0.919, 'meanEm': 0.9, 'maxEm': 0.931, 'adpFm': 0.798, 'meanFm': 0.8, 'maxFm': 0.826}
class SINet_Simp_RFDCT_Dec2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_Simp_RFDCT_Dec2, self).__init__()

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


## rename SINet_AttentionPDC -> SINet_PDCATT
## tested, train0
# ./Result/SINet_COD10K_AllCam_SINet_AttentionPDC/COD10K_all_cam 
#  {'Smeasure': 0.768, 'wFmeasure': 0.537, 'MAE': 0.051, 'adpEm': 0.826, 'meanEm': 0.816, 'maxEm': 0.871, 'adpFm': 0.616, 'meanFm': 0.641, 'maxFm': 0.68}
# ./Result/SINet_COD10K_AllCam_SINet_AttentionPDC/CAMO 
#  {'Smeasure': 0.69, 'wFmeasure': 0.506, 'MAE': 0.121, 'adpEm': 0.781, 'meanEm': 0.698, 'maxEm': 0.776, 'adpFm': 0.637, 'meanFm': 0.588, 'maxFm': 0.618}
# ./Result/SINet_COD10K_AllCam_SINet_AttentionPDC/CHAMELEON 
#  {'Smeasure': 0.858, 'wFmeasure': 0.714, 'MAE': 0.045, 'adpEm': 0.902, 'meanEm': 0.872, 'maxEm': 0.917, 'adpFm': 0.778, 'meanFm': 0.774, 'maxFm': 0.814}
# tested, train1
# [INFO] => [2021-04-14 21:27:16] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_AttentionPDC/COD10K_all_cam]
# {'Smeasure': 0.768, 'wFmeasure': 0.588, 'MAE': 0.047, 'adpEm': 0.83, 'meanEm': 0.826, 'maxEm': 0.87, 'adpFm': 0.621, 'meanFm': 0.647, 'maxFm': 0.676}
# [INFO] => [2021-04-14 21:27:44] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_AttentionPDC/CAMO]
# {'Smeasure': 0.675, 'wFmeasure': 0.51, 'MAE': 0.12, 'adpEm': 0.765, 'meanEm': 0.689, 'maxEm': 0.769, 'adpFm': 0.631, 'meanFm': 0.579, 'maxFm': 0.608}
# [INFO] => [2021-04-14 21:27:57] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_AttentionPDC/CHAMELEON]
# {'Smeasure': 0.853, 'wFmeasure': 0.748, 'MAE': 0.043, 'adpEm': 0.905, 'meanEm': 0.883, 'maxEm': 0.926, 'adpFm': 0.778, 'meanFm': 0.779, 'maxFm': 0.812}
class SINet_PDCATT(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_PDCATT, self).__init__()

        self.resnet = ResNet_2Branch()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf_low_sm = RF(320, channel)
        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_SMATT(channel)

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





## rename SINet_RedF -> SINet_SImpSingle_RFDCT_Dec4
# ./Result/SINet_COD10K_AllCam_SINet_RedF/COD10K_all_cam 
#  {'Smeasure': 0.766, 'wFmeasure': 0.592, 'MAE': 0.046, 'adpEm': 0.84, 'meanEm': 0.823, 'maxEm': 0.87, 'adpFm': 0.632, 'meanFm': 0.65, 'maxFm': 0.676}
# ./Result/SINet_COD10K_AllCam_SINet_RedF/CAMO 
#  {'Smeasure': 0.668, 'wFmeasure': 0.5, 'MAE': 0.12, 'adpEm': 0.766, 'meanEm': 0.673, 'maxEm': 0.77, 'adpFm': 0.629, 'meanFm': 0.564, 'maxFm': 0.605}
# ./Result/SINet_COD10K_AllCam_SINet_RedF/CHAMELEON 
#  {'Smeasure': 0.856, 'wFmeasure': 0.764, 'MAE': 0.042, 'adpEm': 0.913, 'meanEm': 0.889, 'maxEm': 0.928, 'adpFm': 0.797, 'meanFm': 0.798, 'maxFm': 0.826}
class SINet_SimpSingle_RFDCT_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_RFDCT_Dec4, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = nn.Sequential( RF(3584, channel), DCT_Conv(channel, 8, 44, conv_final=True)) 
        self.rf3_sm = nn.Sequential( RF(3072, channel), DCT_Conv(channel, 8, 22, conv_final=True)) 
        self.rf4_sm = nn.Sequential( RF(2048, channel), DCT_Conv(channel, 8, 11, conv_final=True)) 
        self.pdc_sm = PDC_IM(channel)

        self.rf2_im = nn.Sequential(
            RF(512, channel),
            DCT_Conv(channel, 8, 44, conv_final=True),
        ) 
        self.rf3_im = nn.Sequential(
            RF(1280, 256),
            RF(256, channel),
            DCT_Conv(channel, 8, 22, conv_final=True),
        ) 
        self.rf4_im = nn.Sequential(
            RF(2560, 512),
            RF(512, channel),
            DCT_Conv(channel, 8, 11, conv_final=True),
        ) 
        self.pdc_im = PDC_IM(channel)

        self.dec1 = Dec4(256, channel, 256//4)

        self.upsample_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.SA = SA()

    def forward(self, x):
        _, _, x1, x2, x3, x4 = self.resf(x)
        x2_sm, x3_sm, x4_sm = x2, x3, x4

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
        x2_sa = self.SA(camouflage_map_sm.sigmoid(), x2_sm)    # (512, 44, 44)

        x3_im = torch.cat([
            x3_sm,
            self.downSample((1-camouflage_map_sm.sigmoid()).repeat(1,256,1,1)),
        ], dim=1) # 1024+256, 22, 22
        
        x4_im = torch.cat([
            x4_sm,
            self.downSample(self.downSample((1-camouflage_map_sm.sigmoid()).repeat(1,512,1,1))),
        ], dim=1) # 2048 + 512, 11, 11
            
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



## rename SINet_Simp5 -> SINet_SimpSingle_MTuneRF
## tested, train1
# ./Result/SINet_COD10K_AllCam_SINet_Simp5/COD10K_all_cam 
#  {'Smeasure': 0.763, 'wFmeasure': 0.573, 'MAE': 0.048, 'adpEm': 0.836, 'meanEm': 0.812, 'maxEm': 0.863, 'adpFm': 0.622, 'meanFm': 0.642, 'maxFm': 0.67}
# ./Result/SINet_COD10K_AllCam_SINet_Simp5/CAMO 
#  {'Smeasure': 0.673, 'wFmeasure': 0.508, 'MAE': 0.12, 'adpEm': 0.768, 'meanEm': 0.685, 'maxEm': 0.765, 'adpFm': 0.625, 'meanFm': 0.578, 'maxFm': 0.614}
# ./Result/SINet_COD10K_AllCam_SINet_Simp5/CHAMELEON 
#  {'Smeasure': 0.862, 'wFmeasure': 0.759, 'MAE': 0.043, 'adpEm': 0.914, 'meanEm': 0.892, 'maxEm': 0.938, 'adpFm': 0.793, 'meanFm': 0.799, 'maxFm': 0.833}
class SINet_SimpSingle_MTuneRF(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTuneRF, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune(512, 512//4, 512//4, scale_factor=1)
        self.mt3 = MTune(1024, 1024//4, 1024//4, scale_factor=2)
        self.mt4 = MTune(2048, 2048//4, 2048//4, scale_factor=4)
        self.rf2_im = RF(512//4, channel)
        self.rf3_im = RF(1024//4, channel)
        self.rf4_im = RF(2048//4, channel)

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

        x2_im_rf = self.rf2_im(self.mt2(x2_sm, tag))
        x3_im_rf = self.rf3_im(self.mt3(x3_sm, tag))
        x4_im_rf = self.rf4_im(self.mt4(x4_sm, tag))
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)


## rename SINet_Simp6 -> SINet_SimpSingle_MTuneRF_2
## tested, train1
# ./Result/SINet_COD10K_AllCam_SINet_Simp6/COD10K_all_cam 
#  {'Smeasure': 0.763, 'wFmeasure': 0.556, 'MAE': 0.05, 'adpEm': 0.836, 'meanEm': 0.813, 'maxEm': 0.862, 'adpFm': 0.623, 'meanFm': 0.641, 'maxFm': 0.669}
# ./Result/SINet_COD10K_AllCam_SINet_Simp6/CAMO 
#  {'Smeasure': 0.672, 'wFmeasure': 0.498, 'MAE': 0.121, 'adpEm': 0.764, 'meanEm': 0.67, 'maxEm': 0.762, 'adpFm': 0.622, 'meanFm': 0.568, 'maxFm': 0.613}
# ./Result/SINet_COD10K_AllCam_SINet_Simp6/CHAMELEON 
#  {'Smeasure': 0.864, 'wFmeasure': 0.752, 'MAE': 0.042, 'adpEm': 0.914, 'meanEm': 0.897, 'maxEm': 0.937, 'adpFm': 0.8, 'meanFm': 0.803, 'maxFm': 0.834}
class SINet_SimpSingle_MTuneRF_2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTuneRF_2, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune(512, 512//2, 512//4, scale_factor=1)
        self.mt3 = MTune(1024, 1024//2, 1024//4, scale_factor=2)
        self.mt4 = MTune(2048, 2048//2, 2048//4, scale_factor=4)
        self.rf2_im = RF(512//2, channel)
        self.rf3_im = RF(1024//2, channel)
        self.rf4_im = RF(2048//2, channel)

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

        x2_im_rf = self.rf2_im(self.mt2(x2_sm, tag))
        x3_im_rf = self.rf3_im(self.mt3(x3_sm, tag))
        x4_im_rf = self.rf4_im(self.mt4(x4_sm, tag))
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)


## rename SINet_Simp7 -> SINet_SimpSingle_MTuneRF_GA
## tested, train1
# ./Result/SINet_COD10K_AllCam_SINet_Simp7/COD10K_all_cam 
#  {'Smeasure': 0.763, 'wFmeasure': 0.574, 'MAE': 0.048, 'adpEm': 0.831, 'meanEm': 0.816, 'maxEm': 0.87, 'adpFm': 0.62, 'meanFm': 0.639, 'maxFm': 0.671}
# ./Result/SINet_COD10K_AllCam_SINet_Simp7/CAMO 
#  {'Smeasure': 0.666, 'wFmeasure': 0.497, 'MAE': 0.12, 'adpEm': 0.763, 'meanEm': 0.676, 'maxEm': 0.764, 'adpFm': 0.623, 'meanFm': 0.568, 'maxFm': 0.607}
# ./Result/SINet_COD10K_AllCam_SINet_Simp7/CHAMELEON 
#  {'Smeasure': 0.86, 'wFmeasure': 0.753, 'MAE': 0.042, 'adpEm': 0.912, 'meanEm': 0.889, 'maxEm': 0.931, 'adpFm': 0.786, 'meanFm': 0.789, 'maxFm': 0.823}
class SINet_SimpSingle_MTuneRF_GA(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTuneRF_GA, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.GA = GA()
        self.mt2 = MTune(512, 512//2, 512//4, scale_factor=1)
        self.mt3 = MTune(1024, 1024//2, 1024//4, scale_factor=2)
        self.mt4 = MTune(2048, 2048//2, 2048//4, scale_factor=4)
        self.rf2_im = RF(512//2, channel)
        self.rf3_im = RF(1024//2, channel)
        self.rf4_im = RF(2048//2, channel)

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

        tag = self.GA(camouflage_map_sm.sigmoid())

        x2_im_rf = self.rf2_im(self.mt2(x2_sm, tag))
        x3_im_rf = self.rf3_im(self.mt3(x3_sm, tag))
        x4_im_rf = self.rf4_im(self.mt4(x4_sm, tag))
        camouflage_map_im = self.pdc_im(x4_im_rf, x3_im_rf, x2_im_rf)

        return self.upsample_8(camouflage_map_sm), self.upsample_8(camouflage_map_im)


# [INFO] => [2021-04-24 07:58:41] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTunes/COD10K_all_cam]
# {'Smeasure': 0.754, 'wFmeasure': 0.57, 'MAE': 0.049, 'adpEm': 0.829, 'meanEm': 0.818, 'maxEm': 0.862, 'adpFm': 0.61, 'meanFm': 0.63, 'maxFm': 0.654}
# [INFO] => [2021-04-24 07:59:09] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTunes/CAMO]
# {'Smeasure': 0.666, 'wFmeasure': 0.499, 'MAE': 0.12, 'adpEm': 0.765, 'meanEm': 0.681, 'maxEm': 0.763, 'adpFm': 0.625, 'meanFm': 0.567, 'maxFm': 0.608}
# [INFO] => [2021-04-24 07:59:22] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTunes/CHAMELEON]
# {'Smeasure': 0.86, 'wFmeasure': 0.76, 'MAE': 0.041, 'adpEm': 0.908, 'meanEm': 0.887, 'maxEm': 0.92, 'adpFm': 0.789, 'meanFm': 0.79, 'maxFm': 0.819}
class SINet_SimpSingle_MTunes(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTunes, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune(channel, channel, channel//4, scale_factor=1)
        self.mt3 = MTune(channel, channel, channel//4, scale_factor=2)
        self.mt4 = MTune(channel, channel, channel//4, scale_factor=4)
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



# [INFO] => [2021-04-24 03:54:42] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune_Dec4/COD10K_all_cam]
# {'Smeasure': 0.756, 'wFmeasure': 0.579, 'MAE': 0.048, 'adpEm': 0.833, 'meanEm': 0.812, 'maxEm': 0.858, 'adpFm': 0.622, 'meanFm': 0.638, 'maxFm': 0.663}
# [INFO] => [2021-04-24 03:55:11] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune_Dec4/CAMO]
# {'Smeasure': 0.663, 'wFmeasure': 0.5, 'MAE': 0.121, 'adpEm': 0.767, 'meanEm': 0.675, 'maxEm': 0.758, 'adpFm': 0.622, 'meanFm': 0.564, 'maxFm': 0.604}
# [INFO] => [2021-04-24 03:55:24] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune_Dec4/CHAMELEON]
# {'Smeasure': 0.852, 'wFmeasure': 0.753, 'MAE': 0.045, 'adpEm': 0.901, 'meanEm': 0.879, 'maxEm': 0.914, 'adpFm': 0.778, 'meanFm': 0.786, 'maxFm': 0.818}
class SINet_SimpSingle_MTune_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTune_Dec4, self).__init__()

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

        self.dec1 = Dec4(256, channel, 256//4)
        
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


# [INFO] => [2021-04-24 06:49:32] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA_Dec4/COD10K_all_cam]
# {'Smeasure': 0.759, 'wFmeasure': 0.579, 'MAE': 0.048, 'adpEm': 0.837, 'meanEm': 0.817, 'maxEm': 0.865, 'adpFm': 0.622, 'meanFm': 0.639, 'maxFm': 0.669}
# [INFO] => [2021-04-24 06:50:00] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA_Dec4/CAMO]
# {'Smeasure': 0.662, 'wFmeasure': 0.498, 'MAE': 0.122, 'adpEm': 0.759, 'meanEm': 0.674, 'maxEm': 0.768, 'adpFm': 0.632, 'meanFm': 0.568, 'maxFm': 0.618}
# [INFO] => [2021-04-24 06:50:13] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTuneGA_Dec4/CHAMELEON]
# {'Smeasure': 0.855, 'wFmeasure': 0.75, 'MAE': 0.043, 'adpEm': 0.908, 'meanEm': 0.875, 'maxEm': 0.923, 'adpFm': 0.787, 'meanFm': 0.784, 'maxFm': 0.82}
class SINet_SimpSingle_MTuneGA_Dec4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTuneGA_Dec4, self).__init__()

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

        self.dec1 = Dec4(256, channel, 256//4)
        
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


# [INFO] => [2021-04-23 23:01:30] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune3/COD10K_all_cam]
# {'Smeasure': 0.756, 'wFmeasure': 0.574, 'MAE': 0.048, 'adpEm': 0.833, 'meanEm': 0.812, 'maxEm': 0.861, 'adpFm': 0.615, 'meanFm': 0.632, 'maxFm': 0.658}
# [INFO] => [2021-04-23 23:02:00] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune3/CAMO]
# {'Smeasure': 0.678, 'wFmeasure': 0.518, 'MAE': 0.117, 'adpEm': 0.774, 'meanEm': 0.693, 'maxEm': 0.768, 'adpFm': 0.635, 'meanFm': 0.585, 'maxFm': 0.618}
# [INFO] => [2021-04-23 23:02:14] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune3/CHAMELEON]
# {'Smeasure': 0.86, 'wFmeasure': 0.761, 'MAE': 0.041, 'adpEm': 0.912, 'meanEm': 0.888, 'maxEm': 0.929, 'adpFm': 0.79, 'meanFm': 0.792, 'maxFm': 0.822}
class SINet_SimpSingle_MTune3(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTune3, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune2(512, channel, scale_factor=1)
        self.mt3 = MTune2(1024, channel, scale_factor=2)
        self.mt4 = MTune2(2048, channel, scale_factor=4)
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


# [INFO] => [2021-04-23 21:30:14] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune2/COD10K_all_cam]
# {'Smeasure': 0.761, 'wFmeasure': 0.574, 'MAE': 0.048, 'adpEm': 0.818, 'meanEm': 0.821, 'maxEm': 0.862, 'adpFm': 0.603, 'meanFm': 0.632, 'maxFm': 0.66}
# [INFO] => [2021-04-23 21:30:45] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune2/CAMO]
# {'Smeasure': 0.683, 'wFmeasure': 0.526, 'MAE': 0.117, 'adpEm': 0.791, 'meanEm': 0.709, 'maxEm': 0.788, 'adpFm': 0.645, 'meanFm': 0.593, 'maxFm': 0.627}
# [INFO] => [2021-04-23 21:30:59] => [METRIC DONE: ./Result/SINet_COD10K_AllCam_SINet_SimpSingle_MTune2/CHAMELEON]
# {'Smeasure': 0.862, 'wFmeasure': 0.756, 'MAE': 0.043, 'adpEm': 0.904, 'meanEm': 0.893, 'maxEm': 0.934, 'adpFm': 0.777, 'meanFm': 0.788, 'maxFm': 0.82}
class SINet_SimpSingle_MTune2(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32, opt=None):
        super(SINet_SimpSingle_MTune2, self).__init__()

        self.resf = Res_Features()
        self.downSample = nn.MaxPool2d(2, stride=2)

        self.rf2_sm = RF(3584, channel)
        self.rf3_sm = RF(3072, channel)
        self.rf4_sm = RF(2048, channel)
        self.pdc_sm = PDC_IM(channel)

        self.mt2 = MTune(512, channel, scale_factor=1)
        self.mt3 = MTune(1024, channel, scale_factor=2)
        self.mt4 = MTune(2048, channel, scale_factor=4)
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



