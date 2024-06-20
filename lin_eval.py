import logging
import torch
import torch.nn as nn
from torch.optim import SGD
from torchvision import transforms
from torchvision.datasets import ImageFolder, STL10 
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100
from data.data import AdvSimpleTransform, AdvSingleTransform
from models.model import SimSiam

from scheduler import CosineDecayScheduleWithWarmUp, StepSchedule

from utils import fix_random_seed
fix_random_seed(1)

from attack.attack import attack_module
from utils import accuracy

import wandb
from argument import linear_eval_argument as argument_parser
DATA_PATH=''
def train():
    args = argument_parser()
    args.fname = str(args.schedule)+'lr'+str(args.learning_rate)
    if args.rLE:
        args.fname = 'rLE_'+args.fname
    if args.rFT:
        args.fname = 'rFT_'+args.fname
    
    run = wandb.init(
            project='TARO linear evaluation',
            name=args.save_path+'_'+args.fname,
            config=vars(args),
            save_code=True,
            tags=['linear eval', args.target, str(args.learning_rate), str(args.schedule)],
            notes="TARO linear evaluation",
            job_type='train'
        )
    f = open(args.ckpt_path+'/result.txt', 'a')
    f.write(f"file name: {args.fname}\n")
    # dataset
    if args.target == 'cifar10':
        args.num_classes = 10
        train_dataset = CIFAR10(DATA_PATH, train=True, transform=AdvSimpleTransform(args.image_size), download=True)

        valid_dataset = CIFAR10(DATA_PATH, train=False, transform=AdvSingleTransform(args.image_size), download=True)
    
    elif args.target == 'cifar100':
        args.num_classes = 100
        train_dataset = CIFAR100(DATA_PATH, train=True, transform=AdvSimpleTransform(args.image_size), download=True)
        

        valid_dataset = CIFAR100(DATA_PATH, train=False, transform=AdvSingleTransform(args.image_size), download=True)
        
    elif args.target == 'stl10':
        args.num_classes = 10
        train_dataset = STL10(DATA_PATH, split = 'train', transform=AdvSimpleTransform(args.image_size), download=True)
        valid_dataset = STL10(DATA_PATH, split = 'test', transform=AdvSingleTransform(args.image_size), download=True)
    
    else:
        raise NotImplementedError
    train_loader = DataLoader(train_dataset, args.batch_size_per_gpu, shuffle=True, num_workers=args.num_worker, pin_memory=True)
    val_loader = DataLoader(valid_dataset, 100, shuffle=False, num_workers=args.num_worker)
    # model
    model = SimSiam(backbone=args.backbone)
    model_state_dict = torch.load(args.ckpt_path+'/'+args.loading_epoch)
    try:
        model.load_state_dict(model_state_dict['model'])
    except:
        model.load_state_dict(model_state_dict)
    print(f"Pretrain weight loaded")
    feature_dim = model.backbone_output_dim
    model = model.backbone
    model.fc = nn.Linear(in_features=feature_dim, out_features=args.num_classes, bias=True)

    model.to(args.device)
    model = nn.DataParallel(model)
    
    # optimizer
    criterion = nn.CrossEntropyLoss()
    if args.rFT:
        parameter = model.parameters()
        optimizer = SGD(parameter, lr=args.learning_rate)
    else:
        optimizer = SGD(model.module.fc.parameters(), lr=args.learning_rate)

    if args.schedule=='step':
        lr_scheduler = StepSchedule(optimizer, args.max_epochs, len(train_loader), args.learning_rate, internal_step=[90,95,100])
    else:
        lr_scheduler = CosineDecayScheduleWithWarmUp(optimizer, args.max_epochs, len(train_loader), args.learning_rate, 0, args.warmup_epochs, 0)
    
    if args.rLE or args.rFT:
        if args.adv_train == 'AT':
            args.epsilon = args.epsilon/255.
            args.alpha = args.alpha/255.
            train_adversary, _, _, _ = attack_module(args, model, None, criterion=nn.CrossEntropyLoss(), _eval=False, attack_loss=None)
    adversary = attack_module(args, model, None, criterion=nn.CrossEntropyLoss(), _eval=True, attack_loss=None, ssim=False)

    # train
    advbest_acc, best_acc, total_best_acc = 0, 0, 0
    global_step = 0
    for epoch in range(args.max_epochs):

        if args.rFT:
            model.train()
        else:
            model.eval()
            model.module.fc.train()
        max_steps = len(train_loader)
        top1, top3, total_loss, total = 0, 0, 0, 0
        for step, batch in enumerate(train_loader):
            img_batch, y = batch
            x = img_batch[0]
                
            x, y = x.to(args.device), y.to(args.device)

            if args.rLE or args.rFT:
                if args.adv_train == 'AT':
                    x = train_adversary(x, y)

                    optimizer.zero_grad()
                    pred = model(x)

                    loss = criterion(pred, y)
                elif args.adv_train == 'TRADES':
                    from attack import trades_loss
                    loss, pred = trades_loss(model, x, y, optimizer, beta=6.0, distance='Linf')
            else:
                optimizer.zero_grad()
                pred = model(x)
                loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            lr = lr_scheduler.step()

            total_loss += loss.item()

            acc = accuracy(pred, y, topk=(1, 2))
            top1 += acc[0].item()
            top3 += acc[1].item()
            total += 1

            if global_step % 10 == 0:
                print(f"Epoch [{epoch+1}/{args.max_epochs}], Step [{step+1}/{max_steps}], Loss: {loss.item():.4f}")
            wandb.log({'loss': loss.item(), 'epoch': epoch, 'lr': lr}, step=global_step)
            global_step += 1

        print(f"Train [{epoch+1}/{args.max_epochs}], Loss: {total_loss/len(train_loader):.4f}, Top1 acc: {top1/total:.4f}, Top5 acc: {top3/total:.4f}")
        wandb.log({'train_top1': top1/total, 'train_top3': top3/total}, step=global_step)

        model.eval()
        top1, top3, total_loss, total = 0, 0, 0, 0
        with torch.no_grad():
            from tqdm import tqdm
            for step, batch in enumerate(tqdm(val_loader)):
                x, y = batch
                x, y = x.to(args.device), y.to(args.device)
                
                pred = model(x)
                loss = criterion(pred, y).item()
                total_loss += loss

                acc = accuracy(pred, y, topk=(1, 2))
                top1 += acc[0].item()
                top3 += acc[1].item()
                total += 1

            print(f"Valid [{epoch+1}/{args.max_epochs}], Loss: {total_loss/len(val_loader):.4f}, Top1 acc: {top1/total:.4f}, Top5 acc: {top3/total:.4f}")
            f.write(f"Valid [{epoch+1}/{args.max_epochs}], Loss: {total_loss/len(val_loader):.4f}, Top1 acc: {top1/total:.4f}, Top5 acc: {top3/total:.4f}\n")
    
            if top1>best_acc:
                best_acc= top1
                torch.save({'encoder': model.module.state_dict()}, f"{args.ckpt_path}/linear_{args.fname}_best.pt")
        
        advtop1, advtop3, advtotal_loss, advtotal = 0, 0, 0, 1
        if args.fast_eval and epoch<95:
            if (epoch % 10)==0 or epoch==2:
                for step, batch in enumerate(tqdm(val_loader)):
                    x, y = batch
                    x, y = x.to(args.device), y.to(args.device)
                    x = adversary(x, y)
        
                    pred = model(x)
                    loss = criterion(pred, y).item()
                    advtotal_loss += loss

                    acc = accuracy(pred, y, topk=(1, 2))
                    advtop1 += acc[0].item()
                    advtop3 += acc[1].item()
                    advtotal += 1
            
                if advtop1>advbest_acc:
                    advbest_acc= advtop1
                    torch.save({'encoder': model.module.state_dict()}, f"{args.ckpt_path}/linear_{args.fname}_adv_best.pt")
                    
                if ((advtop1 + top1)/2.0)>total_best_acc:
                    total_best_acc = ((advtop1 + top1)/2.0)
                    torch.save({'encoder': model.module.state_dict()}, f"{args.ckpt_path}/linear_{args.fname}_overall_best.pt")
                wandb.log({'advtop1': advtop1/advtotal, 'advtop3': advtop3/advtotal}, step=global_step)

        else:
            for step, batch in enumerate(tqdm(val_loader)):
                x, y = batch
                x, y = x.to(args.device), y.to(args.device)
                x = adversary(x, y)
    
                pred = model(x)
                loss = criterion(pred, y).item()
                advtotal_loss += loss

                acc = accuracy(pred, y, topk=(1, 2))
                advtop1 += acc[0].item()
                advtop3 += acc[1].item()
                advtotal += 1
            
            if advtop1>advbest_acc:
                advbest_acc= advtop1
                torch.save({'encoder': model.module.state_dict()}, f"{args.ckpt_path}/linear_{args.fname}_adv_best.pt")
                
            if ((advtop1 + top1)/2.0)>total_best_acc:
                total_best_acc = ((advtop1 + top1)/2.0)
                torch.save({'encoder': model.module.state_dict()}, f"{args.ckpt_path}/linear_{args.fname}_overall_best.pt")
            
            wandb.log({'advtop1': advtop1/advtotal, 'advtop3': advtop3/advtotal}, step=global_step)


        print(f"ADV Valid [{epoch+1}/{args.max_epochs}], Loss: {advtotal_loss/len(val_loader):.4f}, Top1 acc: {advtop1/advtotal:.4f}, Top5 acc: {advtop3/advtotal:.4f}")
        f.write(f"ADV Valid [{epoch+1}/{args.max_epochs}], Loss: {advtotal_loss/len(val_loader):.4f}, Top1 acc: {advtop1/advtotal:.4f}, Top5 acc: {advtop3/advtotal:.4f}\n")
    
        torch.save({'encoder': model.module.state_dict()}, f"{args.ckpt_path}/linear_{args.fname}_last.pt")
        wandb.log({'top1': top1/total, 'top3': top3/total}, step=global_step)

    run.finish()
    f.close()

if __name__ == '__main__':
    
    train()
