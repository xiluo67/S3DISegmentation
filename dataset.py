from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_file)
        label_path = os.path.join(self.label_dir, image_file.replace('.jpg', '.label').replace('.png', '.label'))

        # Load image
        image = Image.open(image_path).convert('RGB')

        # Load label
        label = np.fromfile(label_path, dtype=np.uint32)  # Adjust dtype as needed
        label = np.reshape(label, (image.height, image.width))  # Adjust shape based on your data format

        # Convert label to PIL Image
        label = Image.fromarray(label.astype(np.uint8))  # Ensure label is in uint8 format for PIL

        if self.transform:
            image, label = self.transform(image, label)

        return image, label