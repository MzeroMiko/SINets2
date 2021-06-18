import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import imageio
import cv2
from Src.Dataloader import test_dataset


def eval_mae(y_pred, y):
    """
    evaluate MAE (for test or validation phase)
    :param y_pred:
    :param y:
    :return: Mean Absolute Error
    """
    return torch.abs(y_pred - y).mean()


def numpy2tensor(numpy):
    """
    convert numpy_array in cpu to tensor in gpu
    :param numpy:
    :return: torch.from_numpy(numpy).cuda()
    """
    return torch.from_numpy(numpy).cuda()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='network name')
    parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
    parser.add_argument('--model_path', type=str, default='./Snapshot/TransSINet_COD10K_default/SINet_40.pth')
    parser.add_argument('--test_save', type=str, default='./Result/TransSINet_COD10K_default/')
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--mask_root', type=str, default='./Dataset/TestDataset/')
    opt = parser.parse_args()
    return opt


def test(opt):
    torch.cuda.set_device(opt.gpu)

    EPS = 1e-8
    model = opt.net(channel=32).cuda()
    model.load_state_dict(torch.load(opt.model_path))
    model.eval()

    for dataset in ['CAMO', 'CHAMELEON', 'COD10K_all_cam']:
        print(f'[INFO: using Dataset {dataset}]')
        # print('-' * 30)
        save_path = opt.test_save + dataset + '/'
        os.makedirs(save_path, exist_ok=True)
        # NOTES:
        #  if you plan to inference on your customized dataset without grouth-truth,
        #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
        #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
        #  the grouth-truth map is unnecessary actually.
        test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Imgs/'.format(dataset),
                                gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                                testsize=opt.testsize)
        for iteration in range(test_loader.size):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + EPS)
            image = image.cuda()
            cam = model(image)[-1]
            cam = F.interpolate(cam, size=gt.shape, mode='bilinear', align_corners=True)
            cam = cam.sigmoid().data.cpu().numpy().squeeze()
            cam = (cam - cam.min()) / (cam.max() - cam.min() + EPS)
            imageio.imwrite(save_path+name, (cam * 255).astype(np.uint8))
            # mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
            # print(f"[Eval-Test] Dataset: {dataset}, Image: {name} ({iteration + 1}/{test_loader.size}), MAE: {mae}")

    print("[Congratulations! Testing Done]")


if __name__ == "__main__":
    opt = get_args()
    if opt.name == '':
        print('please specific a network name')
    else:
        from Src.SINet import SINet_ResNet50 as __SINet
        opt.net = __SINet
        test(opt)
