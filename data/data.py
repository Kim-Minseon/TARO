from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
import cv2
import random
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class SupervisedTransform():
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        x = self.transform(x)
        return x

class AdvSelfTransform():
    def __init__(self, image_size):
        p_blur = 0.5 if image_size > 32 else 0
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
                transforms.RandomResizedCrop(32, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                            saturation=0.2, hue=0.1)],
                    p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
        ])

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

class AdvSimpleTransform():
    def __init__(self, image_size):
        self.val_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

        self.train_transform = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __call__(self, x, adv=False):
        x1 = self.train_transform(x)
        x2 = self.val_transform(x)
        return x1, x2

class AdvSingleTransform():
    def __init__(self, image_size):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __call__(self, x, adv=False):
        x = self.transform(x)
        return x