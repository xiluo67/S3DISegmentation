import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import albumentations.pytorch as A_pytorch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
# Transform
import albumentations as A
# Visualize the testing results
import matplotlib.pyplot as plt
import cv2


torch.cuda.empty_cache()


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

def get_transforms():
    return A.Compose([
        A.HorizontalFlip(),
        A.RandomRotate90(),
        A.OneOf([
            A.RandomBrightnessContrast(),
            A.HueSaturationValue()
        ], p=0.3),
        A.Resize(height=512, width=512, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A_pytorch.ToTensorV2()  # Ensure correct import and usage
    ], p=1.0, is_check_shapes=False)


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
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            # print("image shape:", {images.shape})
            outputs = model(images)
            # Check if outputs is an OrderedDict and extract logits
            if isinstance(outputs, dict):
                logits = outputs['out']
            else:
                logits = outputs

            # print(outputs.shape)
            # print(masks.shape)
            loss = criterion(logits, masks.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # Calculate training accuracy
            _, predicted = torch.max(logits, 1)
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
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)

                if isinstance(outputs, dict):
                    logits = outputs['out']
                else:
                    logits = outputs

                loss = criterion(logits, masks.long())
                val_loss += loss.item() * images.size(0)

                # Calculate validation accuracy
                _, predicted = torch.max(logits, 1)
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
image_folder = '/home/xi/repo/research_2/SP/image/'
mask_folder = '/home/xi/repo/research_2/SP/label/'
# image_folder = '/media/rosie/KINGSTON/Gen_image/PP/image/'
# mask_folder = '/media/rosie/KINGSTON/Gen_image/PP/label/'

train_files, val_files, test_files = split_dataset(image_folder, mask_folder)

# Create datasets
train_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=train_files, transform=get_transforms())
val_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=val_files, transform=get_transforms())
test_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=test_files, transform=get_transforms())

# Create DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=8)
sample = train_dataset[0]
# print("Image shape in dataset:", sample['image'].shape)  # Should print: torch.Size([3, 224, 224])
# print("Mask shape:", sample['mask'].shape)   # Should be consistent with the number of classes
num_classes = 14  # Example number of classes

# get model
from testing_group_1 import *
from testing_group_2 import *
from testing_group_3 import *



train = 0
if train:
    # model = UNet(num_classes=num_classes).to(device)
    # model = DeepLabV3_Pretrained(num_classes=14)
    model = SegFormerPretrained(num_classes=14).to(device)
    model = nn.DataParallel(model)  # Wrap the model with DataParallel
    model.to(device)  # Move the model to GPU
    # model = CustomDPTModel(num_classes=14).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.0e-4)
#     from ranger21 import Ranger21
#
#     optimizer = optimizer = Ranger21(
#     model.parameters(),
#     lr=1e-4,
#     num_epochs=200,
#     num_batches_per_epoch=len(train_dataloader)
# )

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
    # model = SegFormerPretrained(num_classes=14).to(device)
    # model = UNet(num_classes=num_classes).to(device)
    model = SegFormerPretrained(num_classes=14).to(device)
    # model = DeepLabV3_Pretrained(num_classes=14)
    model = nn.DataParallel(model)  # Wrap the model with DataParallel
    model.to(device)  # Move the model to GPU
    model.load_state_dict(torch.load('./log/Dataset1_SegF_SP.pth'))
    model.eval()

    # To store results
    dice_scores = []
    iou_scores = []
    accuracy_scores = []

    # Iterate through the dataloader
    for batch_index, batch in enumerate(test_dataloader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)  # Shape: [batch_size, height, width]

        # Get model predictions
        outputs = model(images)
        if isinstance(outputs, dict):
            logits = outputs['out']
        else:
            logits = outputs
        # Convert outputs to class predictions
        preds = torch.argmax(logits, dim=1)  # Shape: [batch_size, height, width]

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

            # Calculate accuracy
            correct_pixels = torch.sum(pred_cls * mask_cls)
            total_pixels = torch.sum(mask_cls)
            accuracy = correct_pixels / (total_pixels + 1e-8)  # Avoid division by zero

            # Append scores
            dice_scores.append(dice.item())
            iou_scores.append(iou.item())
            accuracy_scores.append(accuracy.item())

    # Average Dice, IoU, and accuracy scores across all classes
    avg_dice = sum(dice_scores) / (len(dice_scores) if dice_scores else 1)
    avg_iou = sum(iou_scores) / (len(iou_scores) if iou_scores else 1)
    avg_accuracy = sum(accuracy_scores) / (len(accuracy_scores) if accuracy_scores else 1)

    print(f"Average Dice Score: {avg_dice:.4f}")
    print(f"Average IoU Score: {avg_iou:.4f}")
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
def plot_results(images, masks, preds, num_samples=2):
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
                logits = preds['out']
            else:
                logits = preds
        preds = torch.argmax(logits, dim=1)  # Assuming the output is logits
        return images, masks, preds

# Fetch a batch and plot
images, masks, preds = get_val_batch(val_dataloader)
plot_results(images, masks, preds)