from CNN import *
# Split the dataset
image_folder = '/home/xi/repo/conference/PP_Dataset1/image/'
mask_folder = '/home/xi/repo/conference/PP_Dataset1/label/'

train_files, val_files = split_dataset(image_folder, mask_folder)
test_files = [f for f in os.listdir('/home/xi/repo/conference/PP_Dataset1/label_test/') if f.endswith('.label')]
print(len(test_files))

# Create datasets
train_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=train_files, transform=get_transforms())
val_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=val_files, transform=get_transforms())
test_dataset = SegmentationDataset(image_folder='/home/xi/repo/conference/PP_Dataset1/image_test/', mask_folder='/home/xi/repo/conference/PP_Dataset1/label_test/',
                                        file_list=test_files, transform=get_transforms())

# Create DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=10, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=10, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=10, drop_last=True)
sample = train_dataset[0]

import gc
torch.cuda.empty_cache()
gc.collect()

num_classes = 14  # Example number of classes
train = 1
if train:
    # model = VGGSegmentation(num_classes=num_classes).to(device)
    # model = UNet(num_classes=num_classes).to(device)
    model = get_pretrianed_unet().to(device)
    # model = DPT.to(device)
    # model = SegFormerPretrained(num_classes=num_classes)
    # model = DeepLabV3
    # model = DeepLabV3_Pretrained(num_classes=num_classes).to(device)
    if torch.cuda.device_count() >= 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=8e-5)

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
    # model = UNet(num_classes=num_classes).to(device)
    model = get_pretrianed_unet().to(device)
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
    model.load_state_dict(torch.load('./log/model_20241203_155248_.pth'))
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

# Fetch a batch and plot
images, masks, preds = get_val_batch(val_dataloader, model)
plot_results(images, masks, preds)