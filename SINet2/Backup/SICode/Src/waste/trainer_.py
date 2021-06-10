import os, pickle, time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


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




