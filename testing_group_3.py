# from testing_group_1 import *

import torch
import torch.nn as nn

"""
This file is used for constructing CNN Arch, test group 3
-----------------------------------------------------------------------------------------------------
Date: 2024.11

"""

num_classes = 14  # Set this to your specific task

# get DeepLabV3
from torchvision import models
import torchvision.models.segmentation as segmentation

class DeepLabV3_Pretrained(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3_Pretrained, self).__init__()
        self.model = segmentation.deeplabv3_resnet101(pretrained=True)
        in_features = self.model.classifier[4].in_channels
        # Replace the classifier head to match num_classes
        self.model.classifier[4] = nn.Conv2d(in_features, num_classes, kernel_size=(1, 1))

        # Freeze all layers except the final classifier (optional for fine-tuning)
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        outputs = self.model(x)
        return outputs


#Get DPT
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDPTModel(nn.Module):
    def __init__(self, num_classes=14):
        super(CustomDPTModel, self).__init__()
        self.model = AutoModel.from_pretrained("intel/dpt-large")
        hidden_size = self.model.config.hidden_size

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_size, out_channels=num_classes, kernel_size=1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state

        batch_size, seq_len, hidden_size = features.size()
        input_height, input_width = x.shape[2], x.shape[3]
        height, width = input_height // 16, input_width // 16

        if seq_len == height * width + 1:
            features = features[:, 1:]
            seq_len = height * width

        features_reshaped = features.view(batch_size, hidden_size, height, width)
        logits = self.segmentation_head(features_reshaped)

        # Resize logits to match the masks' size
        logits_resized = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

        return logits_resized



#Get Segformer
from transformers import SegformerForSemanticSegmentation, SegformerConfig, SegformerImageProcessor
class SegFormerPretrained(nn.Module):
    def __init__(self, num_classes=14):
        super(SegFormerPretrained, self).__init__()
        # Load pretrained SegFormer model
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=num_classes,  # Set number of classes
            ignore_mismatched_sizes=True  # Handle size mismatches
        )

        # Print the full model structure to identify attributes
        print("Full Model Structure:")
        for name, module in self.model.named_modules():
            print(name, module)

        # Identify the component responsible for classification
        # Here you might find a component such as 'segformer' or similar
        self.classifier = None
        for name, module in self.model.named_modules():
            if 'classifier' in name or 'head' in name:
                self.classifier = module
                print(f"Found classifier at: {name}")
                break

        if self.classifier is None:
            raise AttributeError("Unable to find a classification head in the model.")

        # Unfreeze the last few layers of the classification head
        classifier_layers = list(self.classifier.children())
        for layer in classifier_layers[-2:]:  # Unfreeze the last 2 layers
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        outputs = self.model(x)
        logits = outputs['logits']

        # Resize logits to match target size if necessary
        logits_resized = F.interpolate(logits, size=(512, 512), mode='bilinear', align_corners=False)

        return logits_resized


class DeepLabV3Pretrained(nn.Module):
    def __init__(self, num_classes=14):
        super(DeepLabV3Pretrained, self).__init__()
        # Load pretrained DeepLabV3 model with ResNet101 backbone
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=True)

        # Modify the classifier for the desired number of classes
        in_channels = self.model.classifier[4].in_channels
        self.model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))

        # Print the full model structure to identify attributes
        print("Full Model Structure:")
        for name, module in self.model.named_modules():
            print(name, module)

        # Identify the classifier and unfreeze the last two layers
        self.classifier = self.model.classifier
        classifier_layers = list(self.classifier.children())

        # Freeze all layers except the last two layers of the classifier
        for layer in self.classifier.parameters():
            layer.requires_grad = False
        for layer in classifier_layers[-2:]:
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, x):
        outputs = self.model(x)['out']

        # Resize outputs to match target size if necessary
        outputs_resized = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)

        return outputs_resized
