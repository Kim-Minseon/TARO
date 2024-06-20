import torch
import torch.nn as nn
import torch.nn.functional as F

class SimSiamLoss():
    def __init__(self):
        pass

    def __call__(self, p, z):
        z = z.detach()
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        return -(p*z).sum(dim=1).mean()
