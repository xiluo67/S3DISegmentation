# This is a sample Python script.
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import albumentations as A
import albumentations.pytorch as A_pytorch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
# Visualize the testing results
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class VGGSegmentation(nn.Module):
    def __init__(self, num_classes=13):
        super(VGGSegmentation, self).__init__()
        # Pre-trained VGG encoder
        self.encoder = models.vgg16(pretrained=True).features

        # Decoder with upsampling layers
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 14x14

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 28x28

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 56x56

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 112x112

            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # Upsample to 224x224

            nn.Conv2d(16, num_classes, kernel_size=1)  # Final conv layer to get the desired number of classes
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

