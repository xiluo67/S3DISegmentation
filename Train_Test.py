from CNN import *
# Split the dataset
image_folder = '/home/xi/repo/conference/SP_Dataset1/image/'
mask_folder = '/home/xi/repo/conference/SP_Dataset1/label/'

train_files, val_files = split_dataset(image_folder, mask_folder)
test_files = [f for f in os.listdir('/home/xi/repo/conference/SP_Dataset1/label_test/') if f.endswith('.label')]
print(len(test_files))

# Create datasets
train_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=train_files, transform=get_transforms())
val_dataset = SegmentationDataset(image_folder=image_folder, mask_folder=mask_folder, file_list=val_files, transform=get_transforms())
test_dataset = SegmentationDataset(image_folder='/home/xi/repo/conference/SP_Dataset1/image_test/', mask_folder='/home/xi/repo/conference/SP_Dataset1/label_test/',
                                        file_list=test_files, transform=get_transforms())

# Create DataLoaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=10, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=10, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=10, drop_last=True)
sample = train_dataset[0]

import gc
torch.cuda.empty_cache()
gc.collect()

num_classes = 15  # Example number of classes
train = 0
if train:
    # model = UNet(num_classes=num_classes).to(device)
    model = get_pretrianed_unet().to(device)
    # model = Segformer.to(device)
    # model = SegFormerPretrained(num_classes=num_classes)
    if torch.cuda.device_count() >= 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1.2e-4)

    model = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=200, patience=5)
    # Ensure the log directory exists
    log_dir = "./log"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Generate the timestamp
    from datetime import datetime

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save the model with timestamp in the filename
    # model_filename = f"model_{timestamp}_" + ".pth"
    # model_path = os.path.join(log_dir, model_filename)
    model_path = os.path.join(log_dir, "model_UNET_SP1_TL_LR1.2-04.pth")
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")
#     ------------------------------------------------------------------
#     model2 = Segformer.to(device)
#     if torch.cuda.device_count() >= 1:
#         print(f"Let's use {torch.cuda.device_count()} GPUs!")
#         model2 = nn.DataParallel(model2)
#         model2 = model2.cuda()
#
#
#     model2 = train_model(model2, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=200, patience=5)
#     model2_path = os.path.join(log_dir, "model_Seg_PP3_LR1.2-04.pth")
#     torch.save(model2.state_dict(), model2_path)
#
#     print(f"Model saved to {model2_path}")

else:
    # model = UNet(num_classes=num_classes).to(device)
    model = get_pretrianed_unet().to(device)
    # model = Segformer.to(device)
    # model = SegFormerPretrained(num_classes=num_classes)
    if torch.cuda.device_count() >= 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        model = model.to(device)
        model = model.cuda()
    model.load_state_dict(torch.load('./log/model_Seg_TL_SP1_LR1.2-04.pth'))
    model.eval()

    # To store results
    iou_scores = []
    iou_scores_cls = {
        '0': [],
        '1': [],
        '2': [],
        '3': [],
        '4': [],
        '5': [],
        '6': [],
        '7': [],
        '8': [],
        '9': [],
        '10': [],
        '11': [],
        '12': [],
        '13': []
    }
    accuracy_scores = []

    # Iterate through the dataloader
    for batch_index, batch in enumerate(test_dataloader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)  # Shape: [batch_size, height, width]

        # Get model predictions
        outputs = model(images)

        if isinstance(outputs, dict):
            # outputs = outputs['out']
            outputs = outputs['logits']
            outputs = F.interpolate(outputs, size=(512, 512), mode='bilinear', align_corners=False)
        # Convert outputs to class predictions
        preds = torch.argmax(outputs, dim=1)  # Shape: [batch_size, height, width]
        tp = 0
        # Calculate Dice score, IoU, and accuracy for each class
        for cls in range(1, num_classes):
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
            tp += intersection
            iou_cls = intersection / (torch.sum(pred_cls) + torch.sum(mask_cls) - intersection + 1e-8)
            iou_scores_cls[str(cls)].append(iou_cls)

        # Overall Accuracy
        iou = tp / ((preds != 0).sum() + (masks != 0).sum() - tp + 1e-8)
        # iou = tp / (masks.numel() + 1e-8)
        accuracy = (preds == masks).sum().item() / (masks.numel() + 1e-8)

        # Append scores
        iou_scores.append(iou)
        accuracy_scores.append(accuracy)

    # Average IoU, precision, recall, and accuracy scores across all classes
    avg_iou = sum(iou_scores) / (len(iou_scores) if iou_scores else 1)
    avg_accuracy = sum(accuracy_scores) / (len(accuracy_scores) if accuracy_scores else 1)
    for i in range(1, num_classes):
        avg_iou_cl = sum(iou_scores_cls[str(i)]) / (len(iou_scores_cls[str(i)]) if iou_scores_cls[str(i)] else 1)
        print(f"IoU Score: {avg_iou_cl:.4f} for class: {i}")
        
    print(f"Average IoU Score: {avg_iou:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")

# Fetch a batch and plot
images, masks, preds = get_val_batch(val_dataloader, model)
plot_results(images, masks, preds)