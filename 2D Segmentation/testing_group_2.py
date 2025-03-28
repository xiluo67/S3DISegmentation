# from testing_group_1 import *

import torch
"""
This file is used for constructing CNN Arch, test group 2
-----------------------------------------------------------------------------------------------------
Date: 2024.11

"""

num_classes = 15  # Set this to your specific task

# get DeepLabV3
from torchvision import models

DeepLabV3 = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=num_classes)


#get Segformer
from transformers import SegformerForSemanticSegmentation, SegformerConfig

seg_config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Modify the config to change input size and remove pre-trained weights
seg_config.image_size = [256, 256]  # Set input image size to 256x256
seg_config.num_labels = num_classes  # Set number of segmentation classes
# config = SegformerConfig(num_labels=14, image_size=[512, 512])
config = SegformerConfig(
    num_labels=14,                       # Number of classes
    hidden_sizes=[32, 64, 160, 256],     # Encoder hidden sizes
    depths=[2, 2, 2, 2],                 # Number of transformer layers per block
    attention_heads=[1, 2, 5, 8],        # Number of attention heads per block
    image_size=512,                      # Input image size
)
Segformer = SegformerForSemanticSegmentation(config)


#Get DPT
torch.backends.cudnn.benchmark = True
from transformers import DPTForSemanticSegmentation, DPTConfig
# Configure DPT model for training from scratch
dpt_config = DPTConfig.from_pretrained("Intel/dpt-large-ade")

# Modify the config to change input size and remove pre-trained weights
dpt_config.image_size = [256, 256]  # Set input image size to 256x256
dpt_config.num_labels = 14  # Set number of segmentation classes
DPT = DPTForSemanticSegmentation(dpt_config)
DPT.init_weights()  # Uncomment if you want to initialize randomly