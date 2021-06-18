import torch
import torch.nn.functional as F
import numpy as np
import os, argparse, time
import imageio
import cv2
from Src.Dataloader import test_dataset
from Src.data_val import test_dataset as test_dataset_aug

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='network name')
    parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
    parser.add_argument('--model_path', type=str, default='./Snapshot/TransSINet_COD10K_default/SINet.pth')
    parser.add_argument('--test_save', type=str, default='./Result/TransSINet_COD10K_default/')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--mask_root', type=str, default='./Dataset/TestDataset/')
    parser.add_argument('--use_aug', type=int, default=0)
    opt = parser.parse_args()
    return opt


def test(opt):
    torch.cuda.set_device(opt.gpu)
    gettime = lambda :time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    EPS = 1e-8
    model = opt.net(channel=32).cuda()
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()

    for dataset in ['CAMO', 'CHAMELEON', 'COD10K_all_cam']:
        print(f'[INFO] => [{gettime()}] => [Using Dataset {dataset}]')
        save_path = opt.test_save + dataset + '/'
        os.makedirs(save_path, exist_ok=True)
        test_loader = opt.test_dataset(image_root='./Dataset/TestDataset/{}/Imgs/'.format(dataset),
                                gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                                testsize=opt.testsize)
        for _ in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + EPS)
            image = image.cuda()
            cam = model(image)[-1]
            cam = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=opt.align_corners)
            cam = cam.sigmoid().data.cpu().numpy().squeeze()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + EPS)
            imageio.imwrite(save_path+name, (cam * 255).astype(np.uint8))


    print(f'[INFO] => [{gettime()}] => [TEST DONE]')


if __name__ == "__main__":
    opt = get_args()
    if opt.name == '':
        print('please specific a network name')
    else:
        if opt.use_aug != 0:
            ## version 2
            opt.test_dataset = test_dataset_aug
            opt.align_corners = False
            print(f'[INFO] => [Test {opt.name}, Aug=Ture, Align=False]')
        else:
            ## version 1
            opt.test_dataset = test_dataset
            opt.align_corners = True
            print(f'[INFO] => [Test {opt.name}, Aug=False, Align=True]')
        exec("from Src.SINetAtten import " + opt.name+ " as __SINet")
        opt.net = __SINet
        test(opt)
