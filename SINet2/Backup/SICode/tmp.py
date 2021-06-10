from thop import profile, clever_format
"-------------------------------"
# from Src.extra import *
"-------------------------------"
from Src.SINetAtten import *
"-------------------------------"
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse, time
import imageio
import cv2
from Src.Dataloader import test_dataset

def thoptest():
    Models = (
    "SINet_ResNet50",
    "SINet_Simp",
    "SINet_PDCATT",
    "SINet_Simp_PDCN",
    "SINet_Simp_PDCNT",
    "SINet_Simp_PDCA",
    "SINet_RFDCT",
    "SINet_RFFFT",
    "SINet_RFDCTATT",
    "SINet_Simp_RFDCT",
    "SINet_Simp_RFFFT",
    "SINet_Simp_RFDCT_Half",
    "SINet_Simp_RFFFT_Half",
    "SINet_Simp_RFDCT_Half2",
    "SINet_Simp_RFFFT_Half2",
    "SINet_v2",
    "SINet_MinSingle_Fine",
    "SINet_MinSingle_Fine1",
    "SINet_MinSingle_Fine2",
    "SINet_MinSingle_Fine3",
    "SINet_SimpSingle_MTune",
    "SINet_SimpSingle_MTuneRF",
    "SINet_SimpSingle_MTuneRF_2",
    "SINet_SimpSingle_MTuneRF_GA",
    "SINet_SimpSingle_RFDCT_Dec4",
    "SINet_Simp_Dec",
    "SINet_Simp_Dec2",
    "SINet_Simp_Dec3",
    "SINet_Simp_Dec4",
    "SINet_RF2Half",
    "SINet_Simp_RF2Half_Dec",
    "SINet_Simp_RF2DCT_Dec",
    "SINet_Simp_RF2FFT_Dec",
    "SINet_Simp_RF2DCTHalf_Dec",
    "SINet_Simp_RF2FFTHalf_Dec",
    "SINet_Simp_RF2DCTHalf2_Dec",
    "SINet_Simp_RF2FFTHalf2_Dec",
    "SINet_Simp_RF2Half_Dec4",
    "SINet_Simp_RF2DCT_Dec4",
    "SINet_Simp_RF2FFT_Dec4",
    "SINet_Simp_RF2DCTHalf_Dec4",
    "SINet_Simp_RF2FFTHalf_Dec4",
    "SINet_Simp_RF2DCTHalf2_Dec4",
    "SINet_Simp_RF2FFTHalf2_Dec4",
    "SINet_Simp_RFDCT_Dec",
    "SINet_Simp_RFFFT_Dec",
    "SINet_Simp_RFDCTHalf_Dec",
    "SINet_Simp_RFFFTHalf_Dec",
    "SINet_Simp_RFDCTHalf2_Dec",
    "SINet_Simp_RFFFTHalf2_Dec",
    "SINet_Simp_RFDCT_Dec4",
    "SINet_Simp_RFFFT_Dec4",
    "SINet_Simp_RFDCTHalf_Dec4",
    "SINet_Simp_RFFFTHalf_Dec4",
    "SINet_Simp_RFDCTHalf2_Dec4",
    "SINet_Simp_RFFFTHalf2_Dec4",
    )    

    thoplog = open('./thoplog.log', 'w') 
    inp = torch.randn(1,3,352,352)
    for model in Models:
        exec("from Src.SINetAtten import " + model + " as __SINet")
        net = __SINet()
        flops, params = profile(net, inputs=(inp, ))
        params2 = sum(x.numel() for x in net.parameters())
        flops, params, params2 = clever_format([flops, params, params2], "%.3f")
        print(flops, params, params2, model, file=thoplog)
        thoplog.flush()
    thoplog.close()


def dcttest():
    # test DCT1d and DCT2d
    inp = torch.randn(1,3,12,12)
    dct_layer = LinearDCT(in_features=12, type='dct', norm='ortho')
    idct_layer = LinearDCT(in_features=12, type='idct', norm='ortho')
    dctinp = apply_linear_2d(inp, dct_layer)
    idctinp = apply_linear_2d(dctinp, idct_layer)
    dct_layer2 = DCT2d(12)
    idct_layer2 = DCT2d(12, idct=True)
    dctinp2 = dct_layer2(inp)
    idctinp2 = idct_layer2(dctinp2)
    print((dctinp == dctinp2).all(), (idctinp == idctinp2).all())


def vistest(model_name, __SINet, model_path='', save_path='visual', dataset = 'CHAMELEON', ind=75):
    
    if model_path == '':
        model_path = f"./Snapshot/SINet_COD10K_AllCam_{model_name}/SINet.pth"
    model = __SINet(channel=32)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    os.makedirs(save_path, exist_ok=True)

    EPS = 1e-8
    test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Imgs/'.format(dataset),
                                gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                                testsize=352)
    test_loader.index = ind
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + EPS)
    cams = model(image)
    for i in range(len(cams)):
        cam = cams[i]
        filename = os.path.join(save_path, f'{model_name}__{i}__{dataset}__{name}') 
        cam = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + EPS)
        imageio.imwrite(filename, (cam * 255).astype(np.uint8))


# Single_MTune_MinDec 1533MB
# Network 1469MB
# Re_Single_Decs 1587MB
# Single_Decs 1587MB
# Single_MinDecs2 1555MB
# Single_MinDec2s2 1567MB

if __name__ == "__main__":
    # model_name = "SINet_Simp_RF2Half_Dec4"
    # exec("from Src.SINetAttenVis import " + model_name + " as __SINet")
    # vistest(model_name, __SINet)

    inp = torch.randn(1,3,352,352)
    from Src.SINetSingle import Single_MinDec2s2 as __SINet
    net = __SINet()
    flops, params = profile(net, inputs=(inp, ))
    params2 = sum(x.numel() for x in net.parameters())
    flops, params, params2 = clever_format([flops, params, params2], "%.3f")
    print(flops, params, params2)
    for _ in range(300):
        net = net.cuda().eval()
        inp = inp.cuda()
        net(inp)

    
