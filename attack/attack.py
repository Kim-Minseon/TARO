import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import advertorch.attacks as attacks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EvalBN(object):
    def __init__(self, adversary):
        self.adversary = adversary
        self.model = self.adversary.predict

    def __call__(self, *args, **kwargs):
        mode = self.model.training
        self.model.eval()
        return_ = self.adversary(*args, **kwargs)
        self.model.train(mode)
        return return_

class ArtLibWrapper(object):
    def __init__(self, adversary):
        self.adversary = adversary

    def __call__(self, images, labels):
        adv_images = self.adversary.generate(images.cpu(), labels.cpu())
        return torch.tensor(adv_images, device=images.device)

class total_model(nn.Module):
    def __init__(self, encoder, classifier=None, ssim=False):
        super().__init__()
        self.model = encoder
        self.classifier = classifier
        self.ssim = ssim

    def forward(self, inp, inp2=None):
        if self.ssim:
            from torchvision import transforms
            normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            normalize_T = transforms.Compose([transforms.Normalize(*normalize)])
            inp = normalize_T(inp)
            if not (inp2 is None):
                inp2 = normalize_T(inp2)

        if inp2 is None:
            output = self.model(inp)
            if not self.classifier is None:
                output = self.classifier(output)
        else:
            output = self.model(inp, inp2)
            if not self.classifier is None:
                output = self.classifier(output)

        return output



def attack_module(P, model, classifier=None, criterion=None, _eval=False, attack_loss=None, ssim=False):
    
    if attack_loss is None:
        if not (classifier is None):
            model = total_model(model, classifier, ssim).cuda()

    if P.distance == 'Linf':
        _ORD = np.inf
        if _eval:
            _PGD_ALPHA = 8. / 2550.
            _PGD_EPS = 8. / 255.
            _PGD_ITER = 20
        else:
            _PGD_ALPHA = 2. / 255.
            _PGD_EPS = 8. / 255.
            _PGD_ITER = 10
    elif P.distance == 'L2':
        _ORD = 2
        if _eval:
            _PGD_ALPHA = 128. / 2550.
            _PGD_EPS = 128. / 255.
            _PGD_ITER = 20
        else:
            _PGD_ALPHA = 15. / 255.
            _PGD_EPS = 128. / 255.
            _PGD_ITER = 10
    elif P.distance == 'L1':
        _ORD = 1
        if _eval:
            _PGD_ALPHA = 2000. / 2550.
            _PGD_EPS = 2000. / 255.
            _PGD_ITER = 20
        else:
            _PGD_ALPHA = 400. / 255.
            _PGD_EPS = 2000. / 255.
            _PGD_ITER = 10
    else:
        raise NotImplementedError()

    if _eval:
        PGD_ALPHA = _PGD_ALPHA
        PGD_EPS = _PGD_EPS
        PGD_ITER = _PGD_ITER 
    else:
        if P.epsilon is None:
            PGD_EPS = _PGD_EPS
        else:
            PGD_EPS = P.epsilon
        if P.alpha is None:
            PGD_ALPHA = _PGD_ALPHA
        else:
            PGD_ALPHA = P.alpha
        if P.n_iters is None:
            PGD_ITER = _PGD_ITER
        else:
            PGD_ITER = P.n_iters

    adv_kwargs = {'loss_fn': criterion, 'clip_min': 0, 'clip_max': 1}
    if P.adv_method == 'fgsm':
        adv_kwargs.update({'eps': PGD_EPS})
        if P.distance == 'Linf':
            adversary = attacks.GradientSignAttack(model, **adv_kwargs)
        elif P.distance == 'L2':
            adversary = attacks.GradientAttack(model, **adv_kwargs)
        else:
            raise NotImplementedError()
    elif P.adv_method == 'pgd' or P.adv_method == 'pgd_scheduling':
        if 'selfsup' in P.attack_mode or not (attack_loss is None):
            adv_kwargs.update({'eps': PGD_EPS, 'eps_iter': PGD_ALPHA, 'ord': P.distance,
                               'nb_iter': PGD_ITER, 'rand_init': True, 'loss_fn': attack_loss})
            adversary = RepresentationAdv(model, **adv_kwargs)
        elif 'supervised' in P.attack_mode:
            adv_kwargs.update({'eps': PGD_EPS, 'eps_iter': PGD_ALPHA, 'ord': _ORD,
                               'nb_iter': PGD_ITER, 'rand_init': True})
            adversary = attacks.PGDAttack(model, **adv_kwargs)
            print('PGD adversary is loaded')
    elif P.adv_method == 'autoattack':
        from autoattack import AutoAttack
        adversary = AutoAttack(model.eval(), norm='Linf', eps=0.031, version='standard')
    else:
        raise NotImplementedError()

    print('Attack with ', PGD_EPS, PGD_ALPHA, PGD_ITER)
    if _eval:
        return adversary
    else:
        return adversary, PGD_EPS, PGD_ALPHA, PGD_ITER


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)


def l2_norm(x):
    return squared_l2_norm(x).sqrt()


def trades_loss(model,
                x_natural,
                y,
                optimizer,
                step_size=0.003,
                epsilon=0.031,
                perturb_steps=10,
                beta=1.0,
                distance='Linf',
                targeted_attack=0, 
                targeted_image=0,
                targeted_loss='kl'):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    batch_size = len(x_natural)
    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    if distance == 'Linf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                adv_output = model(x_adv)
                clean_output = model(x_natural)
                loss_kl = criterion_kl(F.log_softmax(adv_output, dim=1),
                                        F.softmax(clean_output, dim=1))
                
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    # zero gradient
    optimizer.zero_grad()
    # calculate robust loss
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    adv_output = model(x_adv)
    clean_output = model(x_natural)
    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_output, dim=1),
                                                    F.softmax(clean_output, dim=1))
    loss = loss_natural + beta * loss_robust
    return loss, adv_output



class LatentAttackType():

    def __init__(self, model, eps, eps_iter, nb_iter, clip_min, clip_max, ord='Linf', loss_fn='cos_feat',
                 rand_init=True):

        # Model
        self.predict = model
        # Maximum perturbation
        self.epsilon = eps
        # Movement multiplier per iteration
        self.alpha = eps_iter
        # Minimum value of the pixels
        self.min_val = clip_min
        # Maximum value of the pixels
        self.max_val = clip_max
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = nb_iter
        # The perturbation of epsilon
        self._type = ord
        # loss type
        self.loss_type = loss_fn
        #random_start
        self.random_start = rand_init

    def sim_loss(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        loss = -(p*z).sum(dim=1).mean()
        return loss

    def perturb(self, original_images, target, targeted=True):

        if self.random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.float().cuda()
            x = original_images.float().clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x = x.cuda()
        x.requires_grad = True


        self.predict.eval()

        with torch.enable_grad():
            for _iter in range(self.max_iters):
                (z1, z2), (p1, p2)= self.predict(x, target, feat=False)

                if targeted:
                    if self.loss_type == 'cos_loss':
                        loss = F.cosine_similarity(p1, p2).mean() + F.cosine_similarity(z1, z2).mean()
                    elif self.loss_type == 'sim_loss':
                        loss = 0
                        loss = -self.sim_loss(p1, z2) - self.sim_loss(p2, z1)
                else:
                    if self.loss_type == 'sim_loss':
                        loss = 0
                        loss = self.sim_loss(p1, z2) + self.sim_loss(p2, z1)
                    elif self.loss_type == 'cos_loss':
                        loss = 2.0 - F.cosine_similarity(p1, p2).mean() + F.cosine_similarity(z1, z2).mean()
 
                grad_outputs = None
                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, retain_graph=False)[0]
                
                if self._type == 'Linf':
                    scaled_g = torch.sign(grads.data)

                x.data += self.alpha * scaled_g

                x = torch.clamp(x, self.min_val, self.max_val)
                x = self.project(x, original_images, self.epsilon, self._type)

        self.predict.train()
        return x.detach()

    def project(self, x, original_x, epsilon, _type='Linf'):

        if _type == 'Linf':
            max_x = original_x + epsilon
            min_x = original_x - epsilon

            x = torch.max(torch.min(x, max_x), min_x)
        else:
            raise NotImplementedError

        return x

    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)
