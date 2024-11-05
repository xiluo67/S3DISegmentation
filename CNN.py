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
import cv2
import torch.nn.functional as F
torch.cuda.empty_cache()


"""
This file is used for constructing U-Net CNN and train/save the model
-----------------------------------------------------------------------------------------------------
Date: 2024.11

"""
def visualize_predictions(images, masks, preds, idx):
    plt.figure(figsize=(15, 5))

    # Original Image
    plt.subplot(1, 3, 1)
    img = images[idx].cpu().permute(1, 2, 0).numpy()
    if img.max() > 1:  # Assuming image is in [0, 255] range
        img = img / 255.0  # Normalize to [0, 1]
    plt.imshow(img)
    plt.title("Image")
    plt.axis('off')

    # Ground Truth Mask
    plt.subplot(1, 3, 2)
    mask = masks[idx].cpu().numpy()
    plt.imshow(mask, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 3)
    pred = preds[idx].cpu().numpy()
    plt.imshow(pred, cmap='gray')
    plt.title("Prediction")
    plt.axis('off')

    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
# device = torch.device('cpu')
def split_dataset(image_folder, mask_folder, train_ratio=0.8, val_ratio=0.2, test_ratio=0.01):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # Make sure corresponding mask files exist
    mask_files = [f.replace('.jpg', '.label') for f in image_files if
                  os.path.exists(os.path.join(mask_folder, f.replace('.jpg', '.label')))]

    # Split dataset into training+validation and test sets
    # train_val_files, test_files = train_test_split(mask_files, test_size=test_ratio, random_state=42)

    # Split training+validation set into training and validation sets
    train_files, val_files = train_test_split(mask_files, test_size=val_ratio / (train_ratio + val_ratio),
                                              random_state=42)

    print(len(train_files))
    return train_files, val_files

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
        image_file = mask_file.replace('.label', '.jpg')

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
        # A.HorizontalFlip(),
        # A.RandomRotate90(),
        # A.OneOf([
        #     A.RandomBrightnessContrast(),
        #     A.HueSaturationValue()
        # ], p=0.3),
        A.Resize(height=512, width=512, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A_pytorch.ToTensorV2()  # Ensure correct import and usage
    ], p=1.0)


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


import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes=14):
        super(UNet, self).__init__()

        # Encoder (Downsampling path)
        self.encoder1 = self.conv_block(3, 64)  # Output: [B, 64, 1024, 2048]
        self.encoder2 = self.conv_block(64, 128)  # Output: [B, 128, 512, 1024]
        self.encoder3 = self.conv_block(128, 256)  # Output: [B, 256, 256, 512]
        self.encoder4 = self.conv_block(256, 512)  # Output: [B, 512, 128, 256]
        self.encoder5 = self.conv_block(512, 1024)  # Output: [B, 1024, 64, 128]

        # Decoder (Upsampling path)
        self.upconv5 = self.upconv_block(1024, 512)  # Output: [B, 512, 128, 256]
        self.upconv4 = self.upconv_block(512 + 512, 256)  # Output: [B, 256, 256, 512]
        self.upconv3 = self.upconv_block(256 + 256, 128)  # Output: [B, 128, 512, 1024]
        self.upconv2 = self.upconv_block(128 + 128, 64)   # Output: [B, 64, 1024, 2048]

        # Final conv layer to produce the output segmentation map
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)  # Output: [B, num_classes, 1024, 2048]

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def upconv_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        return block

    def forward(self, x):
        # print(f"Input shape: {x.shape}")

        # Encoding path
        e1 = self.encoder1(x)  # [B, 64, 1024, 2048]
        # print(f"After encoder1: {e1.shape}")
        e2 = self.encoder2(F.max_pool2d(e1, 2))  # [B, 128, 512, 1024]
        # print(f"After encoder2: {e2.shape}")
        e3 = self.encoder3(F.max_pool2d(e2, 2))  # [B, 256, 256, 512]
        # print(f"After encoder3: {e3.shape}")
        e4 = self.encoder4(F.max_pool2d(e3, 2))  # [B, 512, 128, 256]
        # print(f"After encoder4: {e4.shape}")
        e5 = self.encoder5(F.max_pool2d(e4, 2))  # [B, 1024, 64, 128]
        # print(f"After encoder5: {e5.shape}")

        # Decoding path
        d5 = self.upconv5(e5)  # [B, 512, 128, 256]
        # print(f"After upconv5: {d5.shape}")
        d5 = torch.cat((d5, e4), dim=1)  # Concatenate skip connection
        # print(f"After concatenating e4: {d5.shape}")
        d4 = self.upconv4(d5)  # [B, 256, 256, 512]
        # print(f"After upconv4: {d4.shape}")
        d4 = torch.cat((d4, e3), dim=1)  # Concatenate skip connection
        # print(f"After concatenating e3: {d4.shape}")
        d3 = self.upconv3(d4)  # [B, 128, 512, 1024]
        # print(f"After upconv3: {d3.shape}")
        d3 = torch.cat((d3, e2), dim=1)  # Concatenate skip connection
        # print(f"After concatenating e2: {d3.shape}")
        d2 = self.upconv2(d3)  # [B, 64, 1024, 2048]
        # print(f"After upconv2: {d2.shape}")
        d2 = torch.cat((d2, e1), dim=1)  # Concatenate skip connection
        # print(f"After concatenating e1: {d2.shape}")

        out = self.final_conv(d2)  # [B, num_classes, 1024, 2048]
        # print(f"Output shape: {out.shape}")

        return out

from testing_group_3 import *
from testing_group_2 import *
def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=25, patience=5):
    early_stopping = EarlyStopping(patience=patience)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    # Initialize interactive mode for non-blocking plots
    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch in train_dataloader:
            if torch.cuda.device_count() >= 1:
                images = batch['image'].to(device).cuda()
                masks = batch['mask'].to(device).cuda()
            else:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

            optimizer.zero_grad()
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            outputs = model(images)

            # print(outputs.shape)
            # print(masks.shape)
            if isinstance(outputs, dict):
                outputs = outputs['out']
                # outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
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

        train_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs} Training Loss: {epoch_loss:.4f} Accuracy: {train_accuracy:.2f}%')

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch in val_dataloader:
                if torch.cuda.device_count() >= 1:
                    images = batch['image'].to(device).cuda()
                    masks = batch['mask'].to(device).cuda()
                else:
                    images = batch['image'].to(device)
                    masks = batch['mask'].to(device)

                outputs = model(images)

                if isinstance(outputs, dict):
                    outputs = outputs['out']
                    outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks.long())
                val_loss += loss.item() * images.size(0)

                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == masks).sum().item()
                total_val += masks.numel()

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = correct_val / total_val * 100

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f'Validation Loss: {val_loss:.4f} Accuracy: {val_accuracy:.2f}%')

        # Update the plots
        ax[0].clear()
        ax[0].plot(range(epoch + 1), train_losses, label='Train Loss')
        ax[0].plot(range(epoch + 1), val_losses, label='Validation Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].set_title('Training and Validation Loss')

        ax[1].clear()
        ax[1].plot(range(epoch + 1), train_accuracies, label='Train Accuracy')
        ax[1].plot(range(epoch + 1), val_accuracies, label='Validation Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy (%)')
        ax[1].legend()
        ax[1].set_title('Training and Validation Accuracy')

        plt.draw()
        plt.pause(0.1)  # Pause to allow plot to update

        # Check early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the final plot
    return model

# Split the dataset
image_folder = '/home/xi/repo/research_2/PP/image/'
mask_folder = '/home/xi/repo/research_2/PP/label/'
# image_folder = '/media/rosie/KINGSTON/Gen_image/PP/image/'
# mask_folder = '/media/rosie/KINGSTON/Gen_image/PP/label/'

train_files, val_files = split_dataset(image_folder, mask_folder)
test_files = [f for f in os.listdir('/home/xi/repo/research_2/PP/label_t/') if f.endswith('.label')]
print(len(test_files))

# Create datasets
train_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=train_files, transform=get_transforms())
val_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=val_files, transform=get_transforms())
test_dataset = SegmentationDataset(image_folder='/home/xi/repo/research_2/PP/image_t/', mask_folder='/home/xi/repo/research_2/PP/label_t/',
                                        file_list=test_files, transform=get_transforms())

# Create DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=10, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=10, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=10, drop_last=True)
sample = train_dataset[0]

num_classes = 14  # Example number of classes
train = 0
if train:
    # model = VGGSegmentation(num_classes=num_classes).to(device)
    model = UNet(num_classes=num_classes).to(device)
    # model = DPT.to(device)
    # model = SegFormerPretrained(num_classes=num_classes)
    # model = DeepLabV3
    # model = DeepLabV3_Pretrained(num_classes=num_classes).to(device)
    if torch.cuda.device_count() >= 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.2e-4)

    model = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=200, patience=5)
    plt.close()
    # Ensure the log directory exists
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate the timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model with timestamp in the filename
    model_filename = f"model_{timestamp}_" + ".pth"
    model_path = os.path.join(log_dir, model_filename)
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")

else:
    model = UNet(num_classes=num_classes).to(device)
    # model = Segformer.to(device)
    # model = SegFormerPretrained(num_classes=num_classes)
    # model = DeepLabV3.to(device)
    # model = SegFormerPretrained(num_classes=num_classes)
    if torch.cuda.device_count() >= 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model = model.to(device)
        model = model.cuda()
    # model = VGGSegmentation(num_classes).to(device)
    # model = UNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('/home/xi/repo/VGG/log/model_20241015_120101_.pth'))
    model.eval()

    # To store results
    dice_scores = []
    iou_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    # Iterate through the dataloader
    for batch_index, batch in enumerate(test_dataloader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)  # Shape: [batch_size, height, width]

        # Get model predictions
        outputs = model(images)

        if isinstance(outputs, dict):
            outputs = outputs['out']
            # outputs = outputs['logits']
            outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
        # Convert outputs to class predictions
        preds = torch.argmax(outputs, dim=1)  # Shape: [batch_size, height, width]

        # Calculate Dice score, IoU, and accuracy for each class
        for cls in range(num_classes):
            pred_cls = (preds == cls).float()  # Binary mask for predictions
            mask_cls = (masks == cls).float()  # Binary mask for ground truth

            # Ensure shapes match before calculation
            if pred_cls.shape != mask_cls.shape:
                print(f"Batch {batch_index} - Shape mismatch for class {cls}:")
                print(f"Pred_cls Shape: {pred_cls.shape}")
                print(f"Mask_cls Shape: {mask_cls.shape}")

                # Resize pred_cls and mask_cls to match shapes if necessary
                pred_cls = F.interpolate(pred_cls.unsqueeze(1), size=mask_cls.shape[1:], mode='bilinear',
                                         align_corners=False).squeeze(1)
                mask_cls = F.interpolate(mask_cls.unsqueeze(1), size=pred_cls.shape[1:], mode='bilinear',
                                         align_corners=False).squeeze(1)

                print(f"Batch {batch_index} - Resized Pred_cls Shape: {pred_cls.shape}")
                print(f"Batch {batch_index} - Resized Mask_cls Shape: {mask_cls.shape}")

            # Calculate Dice score
            intersection = torch.sum(pred_cls * mask_cls)
            dice = (2. * intersection) / (torch.sum(pred_cls) + torch.sum(mask_cls) + 1e-8)
            iou = intersection / (torch.sum(pred_cls) + torch.sum(mask_cls) - intersection + 1e-8)

            # Calculate precision
            true_positives = intersection
            false_positives = torch.sum(pred_cls) - true_positives
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0.0

            # Calculate recall
            false_negatives = torch.sum(mask_cls) - true_positives
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0.0

            # Calculate accuracy
            correct_pixels = torch.sum(pred_cls * mask_cls)
            total_pixels = torch.sum(mask_cls)
            accuracy = correct_pixels / (total_pixels + 1e-8)  # Avoid division by zero

            # Append scores
            dice_scores.append(dice)
            iou_scores.append(iou)
            precision_scores.append(precision)
            recall_scores.append(recall)
            accuracy_scores.append(accuracy)

    # Average Dice, IoU, precision, recall, and accuracy scores across all classes
    avg_dice = sum(dice_scores) / (len(dice_scores) if dice_scores else 1)
    avg_iou = sum(iou_scores) / (len(iou_scores) if iou_scores else 1)
    avg_precision = sum(precision_scores) / (len(precision_scores) if precision_scores else 1)
    avg_recall = sum(recall_scores) / (len(recall_scores) if recall_scores else 1)
    avg_accuracy = sum(accuracy_scores) / (len(accuracy_scores) if accuracy_scores else 1)

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

# Function to convert tensor to numpy array for plotting
def tensor_to_numpy(tensor):
    tensor = tensor.cpu().numpy()
    if tensor.ndim == 4:  # For batch of images: [batch_size, channels, height, width]
        tensor = tensor.transpose(0, 2, 3, 1)  # Convert to [batch_size, height, width, channels]
        return tensor
    elif tensor.ndim == 3:  # For single image or mask: [channels, height, width] or [height, width]
        if tensor.shape[0] in [1, 3]:  # Single channel or RGB image
            return tensor.transpose(1, 2, 0)  # Convert to [height, width, channels]
        else:
            return tensor.transpose(1, 2, 0)  # Convert to [height, width, 1]
    elif tensor.ndim == 2:  # For masks or predictions: [height, width]
        return tensor[..., np.newaxis]  # Add an extra dimension for consistent shape
    else:
        raise ValueError("Unsupported tensor shape")


# Plot images, masks, and predictions
def plot_results(images, masks, preds, num_samples=4):
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))

    # Adjust spacing
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Adjust vertical and horizontal spacing

    plt.subplots_adjust(
        top=0.936,
        bottom=0.033,
        left=0.015,
        right=0.985,
        hspace=0.357,
        wspace=0.0
    )

    for i in range(num_samples):
        # Image
        ax = axes[i, 0]
        ax.imshow(tensor_to_numpy(images[i] * 255), interpolation='none')
        ax.set_title("Image", fontsize=6, fontweight='light')
        ax.axis('off')

        # Mask
        ax = axes[i, 1]
        ax.imshow(tensor_to_numpy(masks[i]), cmap='gray', interpolation='none')
        ax.set_title("Mask", fontsize=6, fontweight='light')
        ax.axis('off')

        # Prediction
        ax = axes[i, 2]
        ax.imshow(tensor_to_numpy(preds[i]), cmap='gray', interpolation='none')
        ax.set_title("Prediction", fontsize=6, fontweight='light')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# Get a batch of data from the validation dataloader
def get_val_batch(dataloader):
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)  # Shape: [batch_size, height, width]

        with torch.no_grad():
            preds = model(images)

            if isinstance(preds, dict):
                preds = preds['out']
                preds = F.interpolate(preds, size=(512, 512), mode='bilinear', align_corners=False)
        preds = torch.argmax(preds, dim=1)  # Assuming the output is logits
        return images, masks, preds

# Fetch a batch and plot
images, masks, preds = get_val_batch(val_dataloader)
plot_results(images, masks, preds)