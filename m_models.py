import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import copy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import multiprocessing as mp
from torchvision import models
from copy import deepcopy
import warnings
import csv
import torchvision.transforms as transforms
import timm
import random
import torch.backends.cudnn as cudnn
from sklearn.metrics import classification_report
import json


class VGGModel(nn.Module):
  def __init__(self, classes, pretrained=False):
    super(VGGModel, self).__init__()
    self.features = models.vgg16(pretrained=pretrained).features  # Use VGG16 features
    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling
    self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, classes)  # 3 output classes
        )

  def forward(self, x):
    x = self.features(x)
    # print(x.shape)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class FineTunedVGG(nn.Module):
  def __init__(self, classes, pretrained=True):
    super(FineTunedVGG, self).__init__()
    self.features = models.vgg16(pretrained=pretrained).features
    for param in self.features.parameters():
      param.requires_grad = False  # Freeze pre-trained layers

    self.avgpool = nn.AdaptiveAvgPool2d((7, 7))  # Global Average Pooling

    self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, classes)  # 3 output classes
        )

  def forward(self, x):
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class ResNetModel(nn.Module):
  def __init__(self, classes, pretrained=False):
    super(ResNetModel, self).__init__()
    resnetf = models.resnet50(pretrained=pretrained)

    self.features = nn.Sequential(*list(resnetf.children())[:-1]) # Use ResNet50 features
    self.avgpool = nn.AdaptiveAvgPool2d((10, 10))
    self.classifier = nn.Sequential(
      nn.Linear(10 * 10, 4096),  # Adjust based on input size
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(4096, classes)
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), 1, x.size(1), 1)
    # print(x.shape)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x

class FineTunedResNet(nn.Module):
  def __init__(self, classes, pretrained=True):
    super(FineTunedResNet, self).__init__()

    # Load pre-trained ResNet50 model
    resnetf = models.resnet50(pretrained=pretrained)

    self.features = nn.Sequential(*list(resnetf.children())[:-1])

    # Freeze pre-trained layers
    for param in self.features.parameters():
      param.requires_grad = False

    # Replace final layer and adjust for grayscale input
    self.avgpool = nn.AdaptiveAvgPool2d((10, 10))  # Global Average Pooling
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    # self.fc = nn.Linear(self.features.fc.in_features, 3)  # Replace final layer
    self.classifier = nn.Sequential(
            nn.Linear(10 * 10, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, classes)  # 3 output classes
        )

  def forward(self, x):
    # Convert grayscale image to 3-channel tensor (assuming single channel)
    # print(x.size(1))
    if x.size(1) == 1:  # Check if input has 1 channel
      x = x.repeat(1, 3, 1, 1)  # Duplicate grayscale channel 3 times
    # print(x.size(1))

    x = self.features(x)
    x = x.view(x.size(0), 1, x.size(1), 1)
    # print(x.shape)

    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    # print(x.shape)
    x = self.classifier(x)
    return x
    
def vit(classes):
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
    vit_model.head = nn.Linear(vit_model.head.in_features, classes)

    for param in vit_model.patch_embed.parameters():
        param.requires_grad = False
    for param in vit_model.blocks.parameters():
        param.requires_grad = False
    
    return vit_model

def n_vit(classes):
    vit_model = timm.create_model("vit_base_patch16_224", pretrained=True)
    vit_model.head = nn.Linear(vit_model.head.in_features, classes)
    
    return vit_model
