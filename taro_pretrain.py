import torch

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100, STL10
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Subset
from data.data import AdvSingleTransform, AdvSelfTransform
from models.model import SimSiam
from loss import SimSiamLoss
from attack.attack import LatentAttackType
from utils import knn_monitor, _calculate_selective_index, CosineDecayScheduleWithWarmUp_SimSiam, CosineDecayScheduleWithWarmUp
            
from argument import pretrain_argument as argument_parser

def train():
    args = argument_parser()

    Path(args.ckpt_path).mkdir(parents=True, exist_ok=True)
    args.learning_rate = args.learning_rate * args.batch_size_per_gpu / 256
    device = torch.device('cuda')
    
    # dataset
    if args.target == 'cifar10':
        train_dataset = CIFAR10(args.data_path, train=True, transform=AdvSelfTransform(args.image_size), download=True)
        valid_dataset = CIFAR10(args.data_path, train=False, transform=AdvSingleTransform(args.image_size), download=True)
              
    elif args.target == 'cifar100':
        train_dataset = CIFAR100(args.data_path, train=True, transform=AdvSelfTransform(args.image_size), download=True)
        valid_dataset = CIFAR100(args.data_path, train=False, transform=AdvSingleTransform(args.image_size), download=True)
        
    train_loader = DataLoader(train_dataset, args.batch_size_per_gpu, shuffle=True, num_workers=args.num_worker)
    valid_loader = DataLoader(valid_dataset, args.batch_size_per_gpu, shuffle=False, num_workers=args.num_worker)
    
    # models
    model = SimSiam(backbone=args.backbone)
    print(model)
    model.to(device)

    if not (args.resume is None):
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict['model'])
        
        print('resuming the model....')
    model = nn.DataParallel(model)

    criterion = SimSiamLoss()
    
    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': args.learning_rate
    }, {
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': args.learning_rate
    }]

    optimizer = SGD(parameters, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    lr_scheduler = CosineDecayScheduleWithWarmUp_SimSiam(optimizer, args.max_epochs, len(train_loader), args.learning_rate, 0, args.warmup_epochs, 0)
    
    if not (args.resume is None):
        state_dict = torch.load(args.resume)
        start_epoch = state_dict['epoch']
        optimizer.load_state_dict(state_dict['optimizer'])
        current_lr = lr_scheduler.reset(len(train_loader)*start_epoch)
        print(start_epoch, current_lr)
        global_step = len(train_loader)*start_epoch
    else:
        start_epoch = 0
        global_step = 0

    args.epsilon = args.epsilon/255.
    args.alpha = args.alpha/255.
    adversary = LatentAttackType(model, 
                    eps=args.epsilon, eps_iter=args.alpha, 
                    nb_iter=args.n_iters, 
                    clip_min=0, clip_max=1, 
                    ord='Linf', 
                    loss_fn=args.attack_loss,
                    rand_init=True)

    best_acc = 0
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        max_steps = len(train_loader)
        for step, batch in enumerate(train_loader):
            batch, _  = batch
            x1, x2 = batch[0], batch[1]
            x1, x2 = x1.to(device), x2.to(device)

            index1, index2 = _calculate_selective_index(model, x1, x2, args.alpha1, args.alpha2)    

            targeted_x = x1[index1]
            targeted_x2 = x2[index2]
            targeted = True
            c_loss, adv_loss, reg_loss = 0, 0, 0

            advx = adversary(x1, targeted_x, targeted=targeted)
            advx2 = adversary(x2, targeted_x2, targeted=targeted)
            
            optimizer.zero_grad()
            
            (z1, z2, z3), (p1, p2, p3) = model(x1, advx, advx2, feat=False)        
            
            c_loss_1 = criterion(p1, z2)
            c_loss_2 = criterion(p2, z1)
            c_loss = (c_loss_1 + c_loss_2) / 2. # t1, adv t1
        
            adv_loss_1 = criterion(p3, z2)
            adv_loss_2 = criterion(p2, z3) #adv t1, adv t2
            adv_loss_3 = criterion(p3, z1)
            adv_loss_4 = criterion(p1, z3) #t1, adv t2
            adv_loss = (adv_loss_1 + adv_loss_2) / 2. + (adv_loss_3 + adv_loss_4) / 2.

            adv_loss = args.w2 * adv_loss

            loss = c_loss + adv_loss
            loss.backward()
            optimizer.step()
            lr = lr_scheduler.step()

            print(f"Epoch [{epoch+1}/{args.max_epochs}], Step [{step+1}/{max_steps}], Loss: {loss.item():.4f}, c_loss/adv_loss: {c_loss.item():.2f}/{adv_loss.item():.2f}]")

            global_step += 1
    
        with torch.no_grad():
            accuracy, knn_total = 0, 0
                
            accuracy = knn_monitor(args, model.module.backbone, valid_loader, valid_loader, device, k=10)
            print(f"KNN [{epoch}/{args.max_epochs}], Acc: {accuracy:.4f}")
            
        print(f"KNN Accuracy: {accuracy}")
        if epoch > 10:  # Not accurate on first few epochs
            if accuracy > best_acc:
                state_dict = {'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc}
                torch.save(state_dict, f"{args.ckpt_path}/best.pt")
                best_acc = accuracy
                print(f"KNN Best Model Saved to weight/best.pt")

        state_dict = {'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch}
        
        torch.save(state_dict, f"{args.ckpt_path}/last.pt")

        if (epoch % 100 == 0) or (epoch >= args.max_epochs-1):
            torch.save(state_dict, f"{args.ckpt_path}/epoch_{epoch}.pt")
        
        print(f"Model saved in {args.ckpt_path}")

if __name__ == '__main__':
    train()

