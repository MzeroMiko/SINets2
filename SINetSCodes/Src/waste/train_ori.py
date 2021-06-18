import torch
import os, argparse, importlib
from Src.Dataloader import get_loader
from Src.trainer_ import trainer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='network name')
    parser.add_argument('--epoch', type=int, default=40,
                        help='epoch number, default=30')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=36,
                        help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352,
                        help='the size of training image, try small resolutions for speed (like 256)')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1,
                        help='decay rate of learning rate per decay step')
    parser.add_argument('--decay_epoch', type=int, default=30,
                        help='every N epochs decay lr')
    parser.add_argument('--gpu', type=int, default=0,
                        help='choose which gpu you use')
    parser.add_argument('--save_epoch', type=int, default=10,
                        help='every N epochs save your trained snapshot')
    parser.add_argument('--save_model', type=str, default='./Snapshot/2020-CVPR-SINet/')
    parser.add_argument('--train_img_dir', type=str, default='./Dataset/TrainDataset/Image/')
    parser.add_argument('--train_gt_dir', type=str, default='./Dataset/TrainDataset/GT/')
    opt = parser.parse_args()
    return opt


def train(opt):
    torch.cuda.set_device(opt.gpu)

    model_SINet = opt.net(channel=32).cuda()
    optimizer = torch.optim.Adam(model_SINet.parameters(), opt.lr)
    loss_func = torch.nn.BCEWithLogitsLoss()
    # loss_func = Weighted_loss()
    # loss_func = wiouloss()
    train_loader = get_loader(opt.train_img_dir, opt.train_gt_dir, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=8)
    total_step = len(train_loader)

    print('-' * 30)
    # print(model_SINet, '\n', '-' * 30)
    print("[Training Dataset INFO]")
    print(f"img_dir: {opt.train_img_dir} \ngt_dir: {opt.train_gt_dir}") 
    print(f"Learning Rate: {opt.lr} \nBatch Size: {opt.batchsize}") 
    print(f"Training Save: {opt.save_model} \ntotal_num: {total_step}")
    print('-' * 30)
    print(f'The model has {sum(x.numel() for x in model_SINet.parameters())} in total')

    trainer(train_loader=train_loader, model=model_SINet, optimizer=optimizer, opt=opt, loss_func=loss_func, total_step=total_step)


if __name__ == "__main__":
    opt = get_args()
    if opt.name == '':
        print('please specific a network name')
    else:
        from Src.SINet import SINet_ResNet50 as __SINet
        # print(__SINet)
        opt.net = __SINet
        train(opt)
    


