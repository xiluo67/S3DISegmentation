from testing_group_1 import *

import torch
num_classes = 14  # Set this to your specific task

# get DeepLabV3
from torchvision import models

DeepLabV3 = models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=num_classes)


#get Segformer
from transformers import SegformerForSemanticSegmentation, SegformerConfig

seg_config = SegformerConfig.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Modify the config to change input size and remove pre-trained weights
seg_config.image_size = [256, 256]  # Set input image size to 256x256
seg_config.num_labels = num_classes  # Set number of segmentation classes

# Initialize SegFormer model with modified configuration (no pre-trained weights)
Segformer = SegformerForSemanticSegmentation(seg_config)
Segformer.init_weights()  # Uncomment if you want to initialize randomly


#Get DPT
from transformers import DPTForSemanticSegmentation, DPTConfig
# Configure DPT model for training from scratch
dpt_config = DPTConfig.from_pretrained("Intel/dpt-large-ade")

# Modify the config to change input size and remove pre-trained weights
dpt_config.image_size = [256, 256]  # Set input image size to 256x256
dpt_config.num_labels = 14  # Set number of segmentation classes
DPT = DPTForSemanticSegmentation(dpt_config)
DPT.init_weights()  # Uncomment if you want to initialize randomly