import torch
import torch.nn.functional as F
import os, argparse, pickle, time
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from Src.Dataloader import get_loader
from Src.data_val import get_loader as get_loader_aug
# -------------------------------------------
# trainer

def trainer(train_loader, model, optimizer, opt, loss_func, total_step):
    print(f'[INFO] => Options: {opt}')
    model.train()
    scaler = GradScaler()
    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)
    
    gettime = lambda :time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    loss_list = dict()
    loss_list['epoch'] = []
    best_loss, best_epoch = 1e8, 0
    # opt.decay_rate = 0.2
    # opt.max_adjust_step =  3
    # opt.adjust_iter = 3
    # opt.exit_iter = 6
    adjust_step = 0
    for epoch_iter in range(1, opt.max_epoch + 1):
        loss_list[str(epoch_iter)] = []

        # train for one epoch 
        for step, data_pack in enumerate(train_loader):
            optimizer.zero_grad()
            images, gts = data_pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            
            losses = []
            with autocast():
                cam = model(images)
                loss_total = 0
                for _cam in cam:
                    losses.append(loss_func(_cam, gts))
                    loss_total += losses[-1]

            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_list[str(epoch_iter)].append([ l.data for l in losses])

            if step % 10 == 0 or step == total_step:
                print(f'[{gettime()}] => [Epoch Num: {epoch_iter}/{opt.max_epoch}] => [Global Step: {step}/{total_step}] => [', end='')
                for i in range(len(losses)): 
                    print(f'Loss_{i}: {losses[i].data:0.4f} ', end='')
                print(']')

        # show epoch losses
        epoch_losses = torch.Tensor(loss_list[str(epoch_iter)]).sum(dim=0) / len(loss_list[str(epoch_iter)])
        loss_list['epoch'].append(list(epoch_losses))
        print(f'[{gettime()}] => [Epoch Num: {epoch_iter}/{opt.max_epoch}] => [', end='')
        for i in range(len(epoch_losses)): 
            print(f'Epoch Loss_{i}: {epoch_losses[i].data:0.4f} ', end='')
        print(']')

        # adjust learning rate
        sum_epoch_losses = sum(epoch_losses)
        if sum_epoch_losses < best_loss:
            best_loss = sum_epoch_losses
            best_epoch = epoch_iter
            torch.save(model.state_dict(), os.path.join(save_path, 'SINet.pth'))
        elif epoch_iter - best_epoch > opt.exit_iter:
            break
        elif epoch_iter - best_epoch > opt.adjust_iter:
            if adjust_step < opt.max_adjust_step:
                adjust_step += 1
                model.load_state_dict(torch.load(os.path.join(save_path, 'SINet.pth')))
                best_epoch = epoch_iter
                print(f'[INFO] => lr adjusted into  {opt.lr * (opt.decay_rate**adjust_step)}')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= opt.decay_rate
            else:
                break

        if (epoch_iter >= opt.min_epoch) and ((epoch_iter - opt.min_epoch) % opt.save_epoch == 0):
            print(f'[INFO] => Model Saved, iter {epoch_iter}.')
            modeldict = torch.load(os.path.join(save_path, 'SINet.pth'))
            torch.save(modeldict, os.path.join(save_path, f'SINet_{epoch_iter}.pth'))

    print(f'[INFO] => train finish, loss dumped.')
    with open(os.path.join(opt.save_model, 'loss.pkl'), 'wb') as f:
        pickle.dump(loss_list, f)

# --------------------------------------------


def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='', help='network name')
    parser.add_argument('--lr', type=float, default=1e-4, help='init learning rate, try `lr=1e-4`')
    parser.add_argument('--batchsize', type=int, default=36, help='training batch size (Note: ~500MB per img in GPU)')
    parser.add_argument('--trainsize', type=int, default=352, help='the size of training image')
    
    parser.add_argument('--gpu', type=int, default=0, help='choose which gpu you use')
    parser.add_argument('--save_model', type=str, default='./Snapshot/2020-CVPR-SINet/')
    parser.add_argument('--train_img_dir', type=str, default='./Dataset/TrainDataset/Image/')
    parser.add_argument('--train_gt_dir', type=str, default='./Dataset/TrainDataset/GT/')

    parser.add_argument('--max_epoch', type=int, default=64, help='max epoch number')
    parser.add_argument('--save_epoch', type=int, default=20, help='save epoch iter')
    parser.add_argument('--min_epoch', type=int, default=1000, help='min epoch number to star save per save_peoch')

    parser.add_argument('--decay_rate', type=float, default=0.2, help='decay rate of learning rate per decay step')
    parser.add_argument('--max_adjust_step', type=int, default=3, help='max steps for learning rate decay')
    parser.add_argument('--adjust_iter', type=int, default=3, help='max no optim iters for learning rate adjust')
    parser.add_argument('--exit_iter', type=int, default=6, help='max no optim iters for exiting training')
    
    parser.add_argument('--lossf', type=int, default=0, help='---')
    parser.add_argument('--use_aug', type=int, default=0)

    opt = parser.parse_args()
    return opt


def train(opt):
    torch.cuda.set_device(opt.gpu)

    model_SINet = opt.net(channel=32).cuda()
    optimizer = torch.optim.Adam(model_SINet.parameters(), opt.lr)
    train_loader = opt.get_loader(opt.train_img_dir, opt.train_gt_dir, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=8)
    total_step = len(train_loader)

    if opt.lossf == 1:
        loss_func = structure_loss
        opt.lossf = "structure_loss"
    # elif opt.lossf == 2:
        # loss_func = Weighted_loss()
        # opt.lossf = "Weighted_loss()"
    # elif opt.lossf == 3:
        # loss_func = wiouloss()
        # opt.lossf = "wiouloss()"
    else:
        loss_func = torch.nn.BCEWithLogitsLoss()
        opt.lossf = "torch.nn.BCEWithLogitsLoss()"
    
    print('-' * 30)
    # print(model_SINet, '\n', '-' * 30)
    print("[Training Dataset INFO]")
    print(f"loss func: {opt.lossf}")
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
        if opt.use_aug != 0:
            ## version 2
            opt.get_loader = get_loader_aug
            print('[INFO] => [Test SINetv2, Aug=Ture]')
        else:
            ## version 1
            opt.get_loader = get_loader
            print('[INFO] => [Test SINetv1, Aug=False]')
        exec("from Src.SINetAtten import " + opt.name+ " as __SINet")
        # print(__SINet)
        opt.net = __SINet
        train(opt)
    


