import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.preactresnet import ResNet18 as resnet18
from torchvision.models import resnet18
from models.wideresnet import WideResNet

class ProjectionHead(nn.Module):
    """ Projection head. """

    def __init__(self, in_channel, hidden_channel=2048, out_channel=2048, cifar10=True):
        super(ProjectionHead, self).__init__()
        self.cifar10 = cifar10

        self.layer1 = nn.Sequential(
            nn.Linear(in_channel, hidden_channel, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=False)
        )
        self.layer2 = nn.Sequential(
                nn.Linear(hidden_channel, hidden_channel, bias=False),
                nn.BatchNorm1d(hidden_channel),
                nn.ReLU(inplace=False)
            )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_channel, out_channel),
            nn.BatchNorm1d(out_channel, affine=False)
        )

        self.layer3[0].bias.requires_grad = False

    def forward(self, x):
        x = self.layer1(x)
        if self.cifar10 is False:
            x = self.layer2(x)
        x = self.layer3(x)
        return x


class PredictionHead(nn.Module):
    
    def __init__(self, in_channel=2048, hidden_channel=512, out_channel=2048):
        super(PredictionHead, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_channel, hidden_channel, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=False)
        )
        self.layer2 = nn.Linear(hidden_channel, out_channel)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ResnetWrapper(nn.Module):
    def __init__(self, resnet):
        super(ResnetWrapper, self).__init__()
        self.resnet = resnet

        self.feature1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1
        )
        self.feature2 = self.resnet.layer2
        self.feature3 = self.resnet.layer3
        self.feature4 = self.resnet.layer4

    def forward(self, x, feat=True):
        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature4 = self.feature4(feature3)
        final = self.resnet(x)

        if feat:
            return feature1, feature2, feature3, feature4, final
        else:
            return final

class WideResnetWrapper(nn.Module):
    def __init__(self, resnet):
        super(WideResnetWrapper, self).__init__()
        self.resnet = resnet

        self.feature1 = self.resnet.conv1
        self.feature2 = self.resnet.layer1
        self.feature3 = self.resnet.layer2
        self.feature4 = self.resnet.layer3

    def forward(self, x, feat=True):
        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature4 = self.feature4(feature3)
        final = self.resnet(x)

        if feat:
            return feature2, feature3, feature4, final
        else:
            return final
            
class observeWrapper(nn.Module):
    def __init__(self, model):
        super(observeWrapper, self).__init__()
        self.model = model

        self.feature1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1
        )
        self.feature2 = self.model.layer2
        self.feature3 = self.model.layer3
        self.feature4 = self.model.layer4

    def forward(self, x, feat=True):
        feature1 = self.feature1(x)
        feature2 = self.feature2(feature1)
        feature3 = self.feature3(feature2)
        feature4 = self.feature4(feature3)
        final = self.model(x)

        if feat:
            return feature1, feature2, feature3, feature4, final
        else:
            return final

class SimSiam(nn.Module):
    def __init__(self, backbone='resnet18'):
        super(SimSiam, self).__init__()

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=False)

            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.backbone.maxpool = nn.Identity()
            self.backbone.fc = nn.Identity()
          
            self.backbone_wrapper = ResnetWrapper(self.backbone)

            self.backbone_output_dim = 512
            self.projector = ProjectionHead(self.backbone_output_dim, cifar10=True)
        
        elif backbone == 'wideresnet':
            self.backbone = WideResNet(28, 10)
            self.backbone.fc = nn.Identity()
            self.backbone_wrapper = WideResnetWrapper(self.backbone)
            self.backbone_output_dim = 640
            
            self.projector = ProjectionHead(self.backbone_output_dim, cifar10=False)
        self.predictor = PredictionHead()

    def forward(self, x1, x2, x3=None, feat=True):
        if feat:
            if x3 is None:
                e1, e2 = self.backbone_wrapper(x1, feat=False), self.backbone_wrapper(x2, feat=False)
                z1, z2 = self.projector(e1), self.projector(e2)
                p1, p2 = self.predictor(z1), self.predictor(z2)

                return (e1, e2), (z1, z2), (p1, p2)

            else:
                e1, e2, e3 = self.backbone_wrapper(x1, feat=False), self.backbone_wrapper(x2, feat=False),  self.backbone_wrapper(x3, feat=False)
                z1, z2, z3 = self.projector(e1), self.projector(e2), self.projector(e3)
                p1, p2, p3 = self.predictor(z1), self.predictor(z2), self.predictor(z3)

                return (e1, e2, e3), (z1, z2, z3), (p1, p2, p3)
        else:
            if x3 is None:
                e1, e2 = self.backbone_wrapper(x1, feat), self.backbone_wrapper(x2, feat)
                z1, z2 = self.projector(e1), self.projector(e2)
                p1, p2 = self.predictor(z1), self.predictor(z2)

                return (z1, z2), (p1, p2)

            else:
                e1, e2, e3 = self.backbone_wrapper(x1, feat), self.backbone_wrapper(x2, feat),  self.backbone_wrapper(x3, feat)
                z1, z2, z3 = self.projector(e1), self.projector(e2), self.projector(e3)
                p1, p2, p3 = self.predictor(z1), self.predictor(z2), self.predictor(z3)

                return (z1, z2, z3), (p1, p2, p3)

if __name__ == '__main__':
    pass
