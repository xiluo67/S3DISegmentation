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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
def split_dataset(image_folder, mask_folder, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    # Make sure corresponding mask files exist
    mask_files = [f.replace('.png', '.label') for f in image_files if
                  os.path.exists(os.path.join(mask_folder, f.replace('.png', '.label')))]

    # Split dataset into training+validation and test sets
    train_val_files, test_files = train_test_split(mask_files, test_size=test_ratio, random_state=42)

    # Split training+validation set into training and validation sets
    train_files, val_files = train_test_split(train_val_files, test_size=val_ratio / (train_ratio + val_ratio),
                                              random_state=42)

    return train_files, val_files, test_files

class SegmentationDataset(Dataset):
    def __init__(self, image_folder, mask_folder, file_list, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mask_file = self.file_list[idx]
        image_file = mask_file.replace('.label', '.png')

        image_path = os.path.join(self.image_folder, image_file)
        mask_path = os.path.join(self.mask_folder, mask_file)

        # Load image and mask
        image = Image.open(image_path).convert('RGB')
        mask = np.loadtxt(mask_path, dtype=np.uint8)  # Load mask as numpy array

        # Convert PIL image to numpy array
        image_np = np.array(image)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image_np, mask=mask)
            image_np, mask = augmented['image'], augmented['mask']

        # print("image_tensor before:", image_np.shape)
        # Convert numpy arrays to tensors and ensure correct dimensions
        image_tensor = torch.from_numpy(image_np.numpy()).float() / 255.0  # [H, W, C] -> [C, H, W]

        mask_tensor = torch.from_numpy(mask.numpy()).long()  # Ensure mask is in correct format
        # print(" mask tensor after:", mask_tensor.shape)
        return {'image': image_tensor, 'mask': mask_tensor}


# Transform
import albumentations as A
def get_transforms():
    return A.Compose([
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue()
        ], p=0.3),
        A.Resize(height=224, width=224, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A_pytorch.ToTensorV2()  # Ensure correct import and usage
    ], p=1.0, is_check_shapes=False)
# def get_transforms():
#     return A.Compose([
#         A.HorizontalFlip(),
#         A.RandomRotate90(),
#         A.OneOf([
#             A.RandomBrightnessContrast(),
#             A.HueSaturationValue()
#         ], p=0.3),
#         A.Resize(height=224, width=224, always_apply=True),
#         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         A.pytorch.ToTensorV2()
#     ], p=1.0, is_check_shapes=False)



class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

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


def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=25, patience=5):
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == masks).sum().item()
            total_train += masks.numel()

        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_accuracy = correct_train / total_train * 100
        print(f'Epoch {epoch + 1}/{num_epochs} Training Loss: {epoch_loss:.4f} Accuracy: {train_accuracy:.2f}%')

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_dataloader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)

                loss = criterion(outputs, masks.long())
                val_loss += loss.item() * images.size(0)

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == masks).sum().item()
                total_val += masks.numel()

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = correct_val / total_val * 100
        print(f'Validation Loss: {val_loss:.4f} Accuracy: {val_accuracy:.2f}%')

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return model

# Split the dataset
image_folder = '/media/rosie/KINGSTON/research/PP/image/'
mask_folder = '/media/rosie/KINGSTON/research/PP/label/'

train_files, val_files, test_files = split_dataset(image_folder, mask_folder)

# Create datasets
train_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=train_files, transform=get_transforms())
val_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=val_files, transform=get_transforms())
test_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=test_files, transform=get_transforms())

# Create DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)
sample = train_dataset[0]
# print("Image shape in dataset:", sample['image'].shape)  # Should print: torch.Size([3, 224, 224])
# print("Mask shape:", sample['mask'].shape)   # Should be consistent with the number of classes
num_classes = 14  # Example number of classes
model = VGGSegmentation(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=65, patience=7)