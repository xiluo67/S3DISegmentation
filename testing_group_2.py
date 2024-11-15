# from testing_group_1 import *

import torch
"""
This file is used for constructing CNN Arch, test group 2
-----------------------------------------------------------------------------------------------------
Date: 2024.11

"""

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

# class SegformerModel:
#     def __init__(self, num_classes, image_size=(256, 256), model_name="nvidia/segformer-b0-finetuned-ade-512-512",
#                  device='cuda'):
#         """
#         Initializes the Segformer model with a custom configuration (from scratch).
#
#         Args:
#             num_classes (int): The number of segmentation classes.
#             image_size (tuple): The input image size. Default is (256, 256).
#             model_name (str): The model to load from Hugging Face. Default is "nvidia/segformer-b0-finetuned-ade-512-512".
#             device (str): The device to run the model on (e.g., 'cuda' or 'cpu').
#         """
#         self.device = device
#
#         # Load configuration without using pretrained weights
#         self.config = SegformerConfig.from_pretrained(model_name)
#
#         # Modify the configuration for input size and number of classes
#         self.config.image_size = list(image_size)
#         self.config.num_labels = num_classes
#         self.config.ignore_mismatched_sizes = True
#
#         # Initialize the SegFormer model with the custom configuration
#         self.model = SegformerForSemanticSegmentation(config=self.config).to(self.device)
#
#         # Ensure the model is trained from scratch by randomly initializing weights
#         self.model.init_weights()  # This ensures the model starts from scratch
#
#     def forward(self, images):
#         """
#         Perform a forward pass with the Segformer model.
#
#         Args:
#             images (torch.Tensor): A batch of input images.
#
#         Returns:
#             torch.Tensor: The model's output.
#         """
#         outputs = self.model(x)
#         logits = outputs['logits']
#
#         # Resize logits to match target size if necessary
#         logits_resized = F.interpolate(logits, size=(256, 256), mode='bilinear', align_corners=False)
#
#         return logits_resized

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