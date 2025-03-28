# this code is for generating the augmentated images
import os
import hashlib
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np

# Function to generate a unique hash for an image
def get_image_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

class add_gaussian_noise:
    def __init__(self, mean=0.0, std=1.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise
    def __repr__(self):
        return f"{self._class_._name_}(mean={self.mean}. std={self.std})"

import random

# Define paths
image_folder = '/home/xi/repo/conference/PP_Dataset1/image_test/'
mask_folder = '/home/xi/repo/conference/PP_Dataset1/label_test/'
# image_folder = '/home/xi/repo/conference/PP/image/'
# mask_folder = '/home/xi/repo/conference/PP/label/'
augmented_image_folder = '/home/xi/repo/conference/PP_Dataset3/image_test/'
augmented_mask_folder = '/home/xi/repo/conference/PP_Dataset3/label_test/'

# Ensure output directories exist
os.makedirs(augmented_image_folder, exist_ok=True)
os.makedirs(augmented_mask_folder, exist_ok=True)
# Load images and masks
image_files = sorted(os.listdir(image_folder))
mask_files = sorted(os.listdir(mask_folder))

# Sanity check: Ensure same number of images and masks
assert len(image_files) == len(mask_files), "Mismatch in number of images and masks"
print(len(image_files))

# Number of augmentations per image
num_augmentations = 12  # To generate 10x dataset

transform = transforms.Compose([
    # transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=(0.3, 1.0)),
    transforms.ToTensor(),
    add_gaussian_noise(mean=0, std=0.0001),
    transforms.ToPILImage()
])

generated_hashes = set()

# Augmentation loop with deduplication
for idx, (image_file, mask_file) in enumerate(zip(image_files, mask_files)):
    image_path = os.path.join(image_folder, image_file)
    mask_path = os.path.join(mask_folder, mask_file)

    image = Image.open(image_path).convert("RGB")
    mask = np.loadtxt(mask_path)
    # mask_pil = Image.fromarray(mask.astype(np.uint8))  # Convert NumPy array to PIL image for transformations

    for aug_idx in range(num_augmentations):
        seed = random.randint(0, 2**32)
        random.seed(seed)
        augmented_image = transform(image)
        random.seed(seed)
        # augmented_mask_pil = transform(mask_pil)

        # Convert augmented mask back to NumPy array
        # augmented_mask = np.array(augmented_mask_pil)
        augmented_mask = mask

        # Check for duplicates
        img_hash = get_image_hash(augmented_image)
        if img_hash in generated_hashes:
            continue
        generated_hashes.add(img_hash)

        # Save augmented image and mask
        augmented_image.save(os.path.join(augmented_image_folder, f'{idx}_{aug_idx}.png'))
        np.savetxt(os.path.join(augmented_mask_folder, f'{idx}_{aug_idx}.label'), augmented_mask)

    # print(f"Augmented dataset saved to {augmented_image_folder} and {augmented_mask_folder}.")
