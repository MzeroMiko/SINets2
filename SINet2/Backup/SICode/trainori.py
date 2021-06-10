import torch
import os, argparse, importlib
from Src.Dataloader import get_loader
import os, pickle, time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


# ---------------------------------------
# trainer.py

def adjust_lr(optimizer, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


def trainer(train_loader, model, optimizer, opt, loss_func, total_step):
    model.train()
    scaler = GradScaler()
    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)
    
    gettime = lambda :time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    loss_list = dict()   
    for epoch_iter in range(1, opt.epoch + 1):
        loss_list[str(epoch_iter)] = []
        adjust_lr(optimizer, epoch_iter, opt.decay_rate, opt.decay_epoch)

        for step, data_pack in enumerate(train_loader):
            optimizer.zero_grad()
            images, gts = data_pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            
            losses = []
            with autocast():
                # cam_sm, cam_im = model(images)
                # loss_sm = loss_func(cam_sm, gts)
                # loss_im = loss_func(cam_im, gts)
                # loss_total = loss_sm + loss_im
                cam = model(images)
                loss_total = 0
                for _cam in cam:
                    losses.append(loss_func(_cam, gts))
                    loss_total += losses[-1]

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_list[str(epoch_iter)].append(losses)
            if step % 10 == 0 or step == total_step:
                # print(f'[{gettime()}] => [Epoch Num: {epoch_iter}/{opt.epoch}] => [Global Step: {step}/{total_step}] => [Loss_s: {loss_sm.data:0.4f} Loss_i: {loss_im.data:0.4f}]')
                print(f'[{gettime()}] => [Epoch Num: {epoch_iter}/{opt.epoch}] => [Global Step: {step}/{total_step}] => [', end='')
                for i in range(len(losses)): 
                    print(f'Loss_{i}: {losses[i].data:0.4f} ', end='')
                print(']')

        lss = torch.Tensor(loss_list[str(epoch_iter)]).sum(dim=0) / len(loss_list[str(epoch_iter)])
        print(f'[{gettime()}] => [Epoch Num: {epoch_iter}/{opt.epoch}] => [', end='')
        for i in range(len(lss)): 
            print(f'Arg Loss_{i}: {lss[i].data:0.4f} ', end='')
        print(']')

        with open(os.path.join(opt.save_model, 'loss.pth'), 'wb') as f:
            pickle.dump({'epoch ' + str(epoch_iter) : loss_list[str(epoch_iter)]}, f)

        if (epoch_iter) % opt.save_epoch == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'SINet_%d.pth' % epoch_iter))

# -----------------------------------------


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
        exec("from Src.SINetAtten import " + opt.name+ " as __SINet")
        # print(__SINet)
        opt.net = __SINet
        train(opt)
    


