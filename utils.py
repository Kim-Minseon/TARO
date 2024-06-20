import random
import numpy as np
import torch
import os
from torch.utils.data.dataset import Dataset


def _similarity_matrix(B, outputs):
    outputs_norm = outputs/(outputs.norm(dim=1).view(B,1) + 1e-8)
    output_matrix = torch.mm(outputs_norm,outputs_norm.transpose(0,1).detach())
        
    return output_matrix

def _score(B, feature, outputs, alpha1=1.0, alpha2=0.5):
    similarity_value = _similarity_matrix(B, feature.view(B, -1))
    eye = torch.eye(len(similarity_value)).cuda()*-1
    similarity_value = similarity_value +eye
    similarity_value = (similarity_value+1.0)/2.0
    
    soft_output = F.softmax(outputs.detach(), dim=1) 
    log_soft_output =  F.log_softmax(outputs.detach(), dim=1)
    entropy_value = (soft_output * log_soft_output).sum(1)
    entropy_value = entropy_value/torch.max(entropy_value)        
    entropy_value = entropy_value.repeat(B, 1)
    
    total_output = alpha1*nn.functional.normalize(similarity_value) + alpha2*nn.functional.normalize(entropy_value) 

    return total_output

def _selective_index(matrix):
    
    _, final_index = torch.max(matrix, 1)
    
    return final_index

def _calculate_selective_index(model, X1, X2, alpha1, alpha2):
    
    (e1, e2), (z1, z2), (p1, p2) = model(X1, X2, feat=True)
    z1 = z1.detach()
    z2 = z2.detach()
    score_matrix_t1 = _score(X1.size(0), e1.detach(), p1.detach(), alpha1, alpha2)
    selective_index_1 = _selective_index(score_matrix_t1)

    score_matrix_t2 = _score(X2.size(0), e2.detach(), p2.detach(), alpha1, alpha2)
    selective_index_2 = _selective_index(score_matrix_t2)
    
    return selective_index_1, selective_index_2



@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res

"""
From https://github.com/PatrickHua/SimSiam/blob/75a7c51362c30e8628ad83949055ef73829ce786/tools/knn_monitor.py
"""

from tqdm import tqdm
import torch.nn.functional as F 
import torch
# code copied from https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=RI1Y8bSImD7N
# test using a knn monitor
def knn_monitor(args, net, memory_data_loader, test_data_loader, epoch, k=200, t=0.1, hide_progress=False):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting', leave=False, disable=hide_progress):
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader, desc='kNN', disable=hide_progress)
        for data, target in test_bar:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            test_bar.set_postfix({'Accuracy':total_top1 / total_num * 100})
    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

    """
    Scheduler
    """
    from abc import *
import torch
import numpy as np

class StepSchedule(object):
    def __init__(self, optimizer, num_epochs, iter_per_epoch, base_lr, internal_step=None):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr

        if internal_step is None:
            if num_epochs == 25:
                self.internal_step = [self.iter_per_epoch*(num_epochs-10),self.iter_per_epoch*(num_epochs-5)]
            elif num_epochs>=100:
                self.internal_step = [self.iter_per_epoch*75, self.iter_per_epoch*95, self.iter_per_epoch*100]

        else:
            self.internal_step = [x*self.iter_per_epoch for x in internal_step]
        self.current_lr = 0
        self.iter = 0

    def step(self):
        if self.iter<self.internal_step[0]:
            lr = self.base_lr
        elif self.iter<self.internal_step[1]:
            lr = self.base_lr * 0.5
        else:
            lr = self.base_lr * 0.1

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.iter += 1
        self.current_lr = lr
        return lr

    @property
    def get_lr(self):
        return self.current_lr

class CosineDecaySchedule(object):
    def __init__(self, optimizer, num_epochs, iter_per_epoch, base_lr, final_lr):
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.final_lr = final_lr

        self.num_steps = self.num_epochs * self.iter_per_epoch
        self.iter = 0
        self.current_lr = 0

    @abstractmethod
    def step(self):
        lr = self.final_lr+0.5*(self.base_lr-self.final_lr) * \
            (1+np.cos(np.pi*self.iter/self.num_steps))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.iter += 1
        self.current_lr = lr
        return lr

    @property
    @abstractmethod
    def get_lr(self):
        return self.current_lr


class CosineDecayScheduleWithWarmUp(CosineDecaySchedule):
    def __init__(self, optimizer, num_epochs, iter_per_epoch, base_lr, final_lr, warmup_epochs, warmup_lr):
        super(CosineDecaySchedule, self).__init__()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

        self.warmup_steps = self.warmup_epochs * self.iter_per_epoch
        self.decay_steps = self.num_epochs * self.iter_per_epoch
        self.iter = 0
        self.current_lr = 0

    def step(self):
        if self.iter < self.warmup_steps:
            lr = self.warmup_lr+(self.base_lr-self.warmup_lr) * \
                (self.iter/self.warmup_steps)
        else:
            lr = self.final_lr+0.5 * \
                (self.base_lr-self.final_lr) * \
                (1+np.cos(np.pi*self.iter/self.decay_steps))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.iter += 1
        self.current_lr = lr
        return lr

    @property
    def get_lr(self):
        return self.current_lr


class CosineDecayScheduleWithWarmUp_SimSiam(CosineDecaySchedule):
    def __init__(self, optimizer, num_epochs, iter_per_epoch, base_lr, final_lr, warmup_epochs, warmup_lr):
        super(CosineDecaySchedule, self).__init__()
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.iter_per_epoch = iter_per_epoch
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr

        self.warmup_steps = self.warmup_epochs * self.iter_per_epoch
        self.decay_steps = self.num_epochs * self.iter_per_epoch
        self.iter = 0
        self.current_lr = 0

    def step(self):
        if self.iter < self.warmup_steps:
            lr = self.warmup_lr+(self.base_lr-self.warmup_lr) * \
                (self.iter/self.warmup_steps)
        else:
            lr = self.final_lr+0.5 * \
                (self.base_lr-self.final_lr) * \
                (1+np.cos(np.pi*self.iter/self.decay_steps))

        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                param_group['lr'] = lr

        self.iter += 1
        self.current_lr = lr
        return lr

    def reset(self, resume_iter):
        self.iter = resume_iter
        if self.iter < self.warmup_steps:
            lr = self.warmup_lr+(self.base_lr-self.warmup_lr) * \
                (self.iter/self.warmup_steps)
        else:
            lr = self.final_lr+0.5 * \
                (self.base_lr-self.final_lr) * \
                (1+np.cos(np.pi*self.iter/self.decay_steps))

        for param_group in self.optimizer.param_groups:
            if param_group['name'] == 'predictor':
                param_group['lr'] = self.base_lr
            else:
                param_group['lr'] = lr

        self.current_lr = lr
        return lr

    @property
    def get_lr(self):
        return self.current_lr

