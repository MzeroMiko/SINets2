import os, pickle, time
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


def trainer(train_loader, model, optimizer, opt, loss_func, total_step):
    model.train()
    scaler = GradScaler()
    save_path = opt.save_model
    os.makedirs(save_path, exist_ok=True)
    
    gettime = lambda :time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

    loss_list = dict()
    loss_list['epoch'] = []
    best_loss, best_epoch = 1e8, 0
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
        # opt.decay_rate = 0.2
        # opt.max_adjust_step =  3
        # opt.adjust_iter = 3
        # opt.exit_iter = 6
        adjust_step = 0        
        sum_epoch_losses = sum(epoch_losses)
        # print(sum_epoch_losses, best_loss, best_epoch)
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
                print(f'[INFO] => lr adjusted into  {opt.lr * (opt.decay_rate**adjust_step)} ({opt.lr},{opt.decay_rate},{adjust_step})')
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= opt.decay_rate
            else:
                break

        try:
            if epoch_iter >= opt.min_epoch and epoch_iter % opt.save_epoch == 0:
                modeldict = torch.load(os.path.join(save_path, 'SINet.pth'))
                torch.save(modeldict, os.path.join(save_path, f'SINet_{epoch_iter}.pth'))
        except:
            pass

    print(f'[INFO] => train finish, loss dumped.')
    with open(os.path.join(opt.save_model, 'loss.pkl'), 'wb') as f:
        pickle.dump(loss_list, f)

