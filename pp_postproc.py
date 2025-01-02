
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
import matplotlib.pyplot as plt

"""
This file is used for back projection process for perspective projection (reference version / not in use)
----------------------------------------------------------------------------------------------------------
Date: 2024.11

"""
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


def compute_2d_acc(image_pred, image_mask):
    num_classes = len(np.unique(image_mask))
    metrics = {
        "accuracy": 0,
        "precision": np.zeros(num_classes),
        "recall": np.zeros(num_classes),
        "IoU": np.zeros(num_classes)
    }

    # Flatten arrays for easier computation
    ground_truth = image_mask.flatten()
    prediction = image_pred.flatten()

    if isinstance(prediction, torch.Tensor):
        prediction = prediction.numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.numpy()

    # Overall accuracy
    metrics["accuracy"] = np.sum(ground_truth == prediction) / ground_truth.size

    overall_TP = 0
    overall_FP = 0
    overall_FN = 0
    # Per-class metrics
    for cls in range(num_classes):
        # True Positive (TP): Predicted cls, Ground Truth cls
        TP = np.sum((prediction == cls) & (ground_truth == cls))
        overall_TP += TP
        # False Positive (FP): Predicted cls, Ground Truth is not cls
        FP = np.sum((prediction == cls) & (ground_truth != cls))
        overall_FP += FP
        # False Negative (FN): Ground Truth cls, Predicted is not cls
        FN = np.sum((prediction != cls) & (ground_truth == cls))
        overall_FN += FN

        # Calculate Precision, Recall, IoU for each class
        metrics["precision"][cls] = TP / (TP + FP) if (TP + FP) > 0 else 0
        metrics["recall"][cls] = TP / (TP + FN) if (TP + FN) > 0 else 0
        metrics["IoU"][cls] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    IoU = overall_TP / (overall_TP + overall_FP + overall_FN) if (overall_TP + overall_FP + overall_FN) > 0 else 0

    print("2D acc is %f", metrics["accuracy"])
    print("2D Overall IoU is %f", IoU)
    print("2D IoU is %f", metrics["IoU"])

class points_with_pred_labels():
    def __init__(self):
        # Dictionary to store points and their associated labels
        self.point_label_dict = {}

    def add_point_with_label(self, point, label):
        # Convert the point to a tuple to use as a dictionary key
        point_key = point  # Assuming point is (x, y, z)
        # Initialize the key if it doesn't exist and append the label
        if point_key not in self.point_label_dict:
            self.point_label_dict[point_key] = []
        self.point_label_dict[point_key].append(label)

    def get_labels_by_point(self, point):
        # Retrieve the labels using the point's (x, y, z) coordinates
        point_key = point
        # Return the list of labels for the given point or an empty list if not found
        return self.point_label_dict.get(point_key, [])

    def most_voted_labels(self):
        # Initialize a list to hold the (x, y, z, most_voted_label) for each point
        points_with_labels = []

        for point_key, label_list in self.point_label_dict.items():
            if not label_list:
                # If there are no labels, append -1 for the label
                points_with_labels.append((point_key, -1))
                continue

            # Count the occurrences of each label
            label_counts = {}
            for label in label_list:
                # Convert the label to an int if it is an ndarray
                if isinstance(label, np.ndarray):
                    label = label.flatten()  # Flatten the array
                    if label.size == 1:  # Ensure it's a single-element array
                        label = label.item()
                    else:
                        label = tuple(label)  # Convert multi-element arrays to tuples
                label_counts[label] = label_counts.get(label, 0) + 1

            # Find the label(s) with the maximum votes
            max_count = max(label_counts.values())
            max_labels = [label for label, count in label_counts.items() if count == max_count]

            # Determine the most voted label or set to -1 if there's a tie
            if len(max_labels) > 1:
                points_with_labels.append((point_key, -1))
            else:
                points_with_labels.append((point_key, max_labels[0]))

        return points_with_labels


def string_key_to_xyz(string_key, precision=20):
    # Split the string key by underscores
    x_str, y_str, z_str = string_key.split('_')
    scale_factor = 10 ** precision
    # Convert the strings back to floats and divide by the scale factor
    x = float(x_str) / scale_factor
    y = float(y_str) / scale_factor
    z = float(z_str) / scale_factor

    return (x, y, z)

def complete_labels_with_knn(k, point_cloud):
    # Project all points
    extended_points = point_cloud_dataset.most_voted_labels()
    extended_points[0] = [string_key_to_xyz(point[0]) for point in extended_points]

    projected_points = [point for point in extended_points if point[-1] != -1]
    unprojected_points = [point for point in extended_points if point[-1] == -1]

    projected_features = [string_key_to_xyz(point[0]) for point in projected_points]  # Extract (x, y, z)
    projected_labels = [point[-1] for point in projected_points]  # Extract label

    unique_classes = np.unique(projected_labels)
    print("Unique classes:", unique_classes)

    unprojected_points_in_cloud = []

    # Apply KNN if there are unprojected points
    if len(unprojected_points) > 0 or len(unprojected_points_in_cloud) > 0:
        # Fit KNN on projected data
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(projected_features, np.array(projected_labels).astype(int))

        # Predict labels for original unprojected points
        unprojected_points = np.array(unprojected_points)

        if len(unprojected_points) > 0:
            unprojected_features = [string_key_to_xyz(p[0]) for p in unprojected_points]
            predicted_labels_for_unprojected = knn.predict(unprojected_features)
            unprojected_points_with_labels = np.hstack(
                (unprojected_points[:, 0:3], predicted_labels_for_unprojected.reshape(-1, 1)))
        else:
            unprojected_points_with_labels = np.array([])

        # Predict labels for unprojected points in the original point cloud
        if len(unprojected_points_in_cloud) > 0:
            unprojected_cloud_features = unprojected_points_in_cloud  # Extract (x, y, z) features
            predicted_labels_for_unprojected_cloud = knn.predict(unprojected_cloud_features)
            unprojected_points_in_cloud_with_labels = np.hstack(
                (unprojected_points_in_cloud, predicted_labels_for_unprojected_cloud.reshape(-1, 1)))
        else:
            unprojected_points_in_cloud_with_labels = np.array([])

        # Combine results
        if len(unprojected_points_with_labels) > 0 and len(unprojected_points_in_cloud_with_labels) > 0:
            all_points = np.vstack(
                (projected_points, unprojected_points_with_labels, unprojected_points_in_cloud_with_labels))
        elif len(unprojected_points_with_labels) > 0:
            all_points = np.vstack((projected_points, unprojected_points_with_labels))
        elif len(unprojected_points_in_cloud_with_labels) > 0:
            all_points = np.vstack((projected_points, unprojected_points_in_cloud_with_labels))
        else:
            all_points = projected_points
    else:
        # If no unprojected points, just return the projected points
        all_points = projected_points

    print(type(all_points))
    return all_points

def postprocess_output(output, target_size=(512, 512)):
    """
    Post-process the model output to ensure it has a shape of target_size.

    Args:
    - output (Tensor): The raw output tensor from the model. Assumes shape [batch_size, num_classes, height, width].
    - target_size (tuple): The desired output size (height, width).

    Returns:
    - np.ndarray: The processed label image of size target_size.
    """
    # Convert output tensor to numpy array
    output = output.squeeze(0).cpu().detach().numpy()  # Remove batch dimension

    # Get the class prediction for each pixel
    if output.ndim == 3:  # If the output is [num_classes, height, width]
        label_image = np.argmax(output, axis=0)  # Shape: [height, width]
    elif output.ndim == 2:  # If the output is [height, width]
        label_image = output
    else:
        raise ValueError("Unexpected output shape: {}".format(output.shape))

    # Ensure the label_image is in the target size (256, 256)
    if label_image.shape != target_size:
        label_image = resize(label_image, target_size, mode='reflect', anti_aliasing=False)

    return label_image

# **********************************************************************************************************
def look_at(camera_position, target, up):
    forward = np.float_(target - camera_position)
    forward /= np.linalg.norm(forward)

    right = np.cross(up, forward)
    right /= np.linalg.norm(right)

    up = np.cross(forward, right)

    rotation_matrix = np.array([
        [right[0], up[0], forward[0], 0],
        [right[1], up[1], forward[1], 0],
        [right[2], up[2], forward[2], 0],
        [0, 0, 0, 1]
    ])

    translation_matrix = np.array([
        [1, 0, 0, -camera_position[0]],
        [0, 1, 0, -camera_position[1]],
        [0, 0, 1, -camera_position[2]],
        [0, 0, 0, 1]
    ])

    view_matrix = rotation_matrix @ translation_matrix
    return view_matrix

def perspective_projection(fov, aspect_ratio, near, far):
    f = 1 / np.tan(fov / 2)
    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])
    return projection_matrix

def project_to_screen(ndc_coords, width, height):
    x_ndc, y_ndc = ndc_coords[:, 0], ndc_coords[:, 1]
    x_screen = (x_ndc + 1) * 0.5 * width
    y_screen = (1 - y_ndc) * 0.5 * height
    ndc_coords[:, 0] = x_screen
    ndc_coords[:, 1] = y_screen
    return ndc_coords

def project_to_screen(ndc_coords, width, height):
    x_ndc, y_ndc = ndc_coords[:, 0], ndc_coords[:, 1]
    x_screen = (x_ndc + 1) * 0.5 * width
    y_screen = (y_ndc + 1) * 0.5 * height
    ndc_coords[:, 0] = x_screen
    ndc_coords[:, 1] = y_screen
    return ndc_coords

# *************************************************************************************
point_cloud = np.loadtxt(
    "/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_1/hallway_2/room_data/hallway_2.txt")
point_label = np.loadtxt(
    "/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_1/hallway_2/room_data/hallway_2.label")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(num_classes=5).to(device)
if torch.cuda.device_count() >= 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)
    model = model.cuda()
model.load_state_dict(torch.load('/home/xi/repo/VGG/log/UNet-Hallway-Singleroom.pth'))
model.eval()

point_cloud_dataset = points_with_pred_labels()
gt_dataset = points_with_pred_labels()

def pack_coordinates(x, y, z, precision=20):

    scale_factor = 10 ** precision

    # Scale x, y, z to integers
    xi = int(-x * scale_factor)
    yi = int(y * scale_factor)
    zi = int(z * scale_factor)

    # Create a unique key by concatenating the scaled integers
    string_key = f"{xi}_{yi}_{zi}"

    return string_key

# *************************************************************************************
def transform_point(point, model_matrix, view_matrix, projection_matrix):
    world_coords = np.dot(model_matrix, point[0:4, :])
    camera_coords = np.dot(view_matrix, world_coords)
    clip_coords = np.dot(projection_matrix, camera_coords)
    ndc_coords = np.vstack(
        ((clip_coords[:3, :] / clip_coords[3, :]), point[4, :], point[5, :], point[6, :], point[7, :]))
    return np.transpose(ndc_coords)

def do_mid_pp_projection(points_3d, label, view, fov_value):
    # Room Dimension
    X = np.max(points_3d[:, 1]) - np.min(points_3d[:, 1])
    Y = np.max(points_3d[:, 2]) - np.min(points_3d[:, 2])
    Z = np.max(points_3d[:, 0]) - np.min(points_3d[:, 0])

    camera_position = np.array([np.mean(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.mean(points_3d[:, 0])])
    up = np.array([0, 1, 0])

    if (view == 0):
        target = np.array([np.min(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.mean(points_3d[:, 0])])
    elif (view == 1):
        target = np.array(
            [np.max(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.mean(points_3d[:, 0])])
    elif (view == 2):
        target = np.array(
            [np.mean(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.min(points_3d[:, 0])])
    elif (view == 3):
        target = np.array(
            [np.mean(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.max(points_3d[:, 0])])
    elif (view == 4):
        target = np.array(
            [np.mean(points_3d[:, 1]), np.max(points_3d[:, 2]), np.mean(points_3d[:, 0])])
        up = np.array([0, 0, 1])
    elif (view == 5):
        target = np.array(
            [np.mean(points_3d[:, 1]), np.min(points_3d[:, 2]), np.mean(points_3d[:, 0])])
        up = np.array([0, 0, 1])
    else:
        return 0

    # Define model matrix (identity for simplicity)
    model_matrix = np.identity(4)

    # Compute view and projection matrices
    view_matrix = look_at(camera_position, target, up)
    projection_matrix = perspective_projection(fov_value, 1, 0.1, far=1000)
    # Define a 3D point in homogeneous coordinates
    points = np.hstack(
        (points_3d[:, 1].reshape(-1, 1), points_3d[:, 2].reshape(-1, 1), points_3d[:, 0].reshape(-1, 1),
         np.ones((points_3d.shape[0], 1)), points_3d[:, 3:], label.reshape(-1, 1)))  # (x, y, z, w)

    # filter out the 3d points behind the camera and looking direction
    forward = np.float_(target - camera_position)
    forward /= np.linalg.norm(forward)
    dot_products = np.dot((points[:, 0:3] - camera_position), forward)
    mask = dot_products > 0
    valid_points = points[mask]

    # Transform the point through the whole pipeline
    new_coords = transform_point(np.transpose(valid_points), model_matrix, view_matrix, projection_matrix)

    # Project to screen coordinates
    screen_coords = project_to_screen(np.array(new_coords), width=512, height=512)

    x_coords = screen_coords[:, 0].astype(np.int32)

    y_coords = screen_coords[:, 1].astype(np.int32)

    colors = screen_coords[:, 3:6].astype(np.uint8)  # Assuming RGB values in uint8 format

    labels = screen_coords[:, 6:].astype(np.uint8)  # Assuming RGB values in uint8 format

    # Calculate the actual dimensions of the image
    width = 512
    height = 512
    img_width = width
    img_height = height

    # Create a blank image with specified dimensions
    image = np.zeros((height, width, 3), dtype=np.uint8)
    proj_l = np.zeros((height, width), dtype=np.uint8)
    # Assign colors to corresponding coordinates within the limits
    valid_indices = (x_coords < img_width) & (y_coords < img_height) & (0 <= x_coords) & (0 <= y_coords)
    image[y_coords[valid_indices], x_coords[valid_indices]] = colors[valid_indices]
    proj_l[y_coords[valid_indices], x_coords[valid_indices]] = labels[valid_indices].reshape(-1)
    # np.savetxt(save_label, proj_l)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if isinstance(image_rgb, np.ndarray):
        image_rgb = Image.fromarray(image_rgb)
    else:
        image_rgb = image_rgb
    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Adjust to match your model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])

    # Apply transformations
    preprocessed_image = transform(image_rgb).unsqueeze(0)  # Add batch dimension


    with torch.no_grad():
        # Ensure the input is a PyTorch tensor
        preprocessed_image = torch.tensor(preprocessed_image, dtype=torch.float32).to(device)
        output = model(preprocessed_image)

        # Handle model outputs
        if isinstance(output, dict):
            logits = output['out']
        else:
            logits = output

    # Convert logits to class predictions
    label_image = torch.argmax(logits, dim=1)  # Shape: [batch_size, height, width]
    label_image = postprocess_output(label_image)
    # To store results
    dice_scores = []
    iou_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []

    for cls in range(5):
        pred_cls = (label_image == cls).astype(np.float32).reshape((1, 512, 512))  # Binary mask for predictions
        mask_cls = (proj_l == cls).astype(np.float32).reshape(pred_cls.shape)  # Binary mask for ground truth
        mask_cls = torch.from_numpy(mask_cls).to(device)
        pred_cls = torch.from_numpy(pred_cls).to(device)
        # Ensure shapes match before calculation
        if pred_cls.shape != mask_cls.shape:
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

    for indice in range(len(valid_indices)):
        if (valid_indices[indice]):
            key = pack_coordinates(valid_points[indice, 2], valid_points[indice, 0], valid_points[indice, 1])
            # key = (valid_points[indice, 2], valid_points[indice, 0], valid_points[indice, 1])
            point_cloud_dataset.add_point_with_label(key, label_image[y_coords[indice], x_coords[indice]])

def get_3d_eval_res(predicted_labels, ground_truth_labels):
    # Determine the unique classes from the ground truth labels
    classes = np.unique(ground_truth_labels)
    num_classes = len(classes)
    print("Unique classes in gt:", classes)
    # Initialize variables to store the metrics
    accuracy = np.mean(predicted_labels == ground_truth_labels)
    class_accuracy = np.zeros(num_classes)
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    iou = np.zeros(num_classes)

    # Initialize accumulators for global metrics
    global_tp = 0
    global_fp = 0
    global_fn = 0
    global_gt_total = 0

    # Create a mapping from class values to indices
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for class_value in classes:
        class_idx = class_to_index[class_value]

        # True positives, false positives, and false negatives for each class
        true_positives = np.sum((predicted_labels == class_value) & (ground_truth_labels == class_value))
        false_positives = np.sum((predicted_labels == class_value) & (ground_truth_labels != class_value))
        false_negatives = np.sum((predicted_labels != class_value) & (ground_truth_labels == class_value))
        total_ground_truth = np.sum(ground_truth_labels == class_value)

        # Class-specific accuracy
        if total_ground_truth > 0:
            class_accuracy[class_idx] = true_positives / total_ground_truth
        else:
            class_accuracy[class_idx] = 0.0

        # Precision: TP / (TP + FP)
        if true_positives + false_positives > 0:
            precision[class_idx] = true_positives / (true_positives + false_positives)
        else:
            precision[class_idx] = 0.0

        # Recall: TP / (TP + FN)
        if true_positives + false_negatives > 0:
            recall[class_idx] = true_positives / (true_positives + false_negatives)
        else:
            recall[class_idx] = 0.0

        # IoU: TP / (TP + FP + FN)
        if true_positives + false_positives + false_negatives > 0:
            iou[class_idx] = true_positives / (true_positives + false_positives + false_negatives)
        else:
            iou[class_idx] = 0.0

        # Accumulate for global metrics
        global_tp += true_positives
        global_fp += false_positives
        global_fn += false_negatives
        global_gt_total += total_ground_truth

        # Print metrics for the current class
        print(f"Class {class_value}:")
        print(f"  Accuracy: {class_accuracy[class_idx]:.4f}")
        print(f"  Precision: {precision[class_idx]:.4f}")
        print(f"  Recall: {recall[class_idx]:.4f}")
        print(f"  IoU: {iou[class_idx]:.4f}")
        print()

    # Compute overall metrics
    if global_gt_total > 0:
        overall_accuracy = global_tp / global_gt_total
    else:
        overall_accuracy = 0.0

    if global_tp + global_fp > 0:
        overall_precision = global_tp / (global_tp + global_fp)
    else:
        overall_precision = 0.0

    if global_tp + global_fn > 0:
        overall_recall = global_tp / (global_tp + global_fn)
    else:
        overall_recall = 0.0

    if global_tp + global_fp + global_fn > 0:
        overall_iou = global_tp / (global_tp + global_fp + global_fn)
    else:
        overall_iou = 0.0

    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")
    return overall_accuracy, overall_precision, overall_recall, overall_iou

def visualize_points_with_colored_labels_open3d(points):
    # Assuming points is an (N, 9) array where:
    # points[:, 0:3] -> xyz coordinates
    # points[:, 6]   -> labels (l)

    xyz = points[:, 0:3]  # Extract xyz coordinates
    labels = points[:, -1].astype(int)  # Extract labels and convert to integers

    # Create a colormap with distinct colors for each label
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)

    # Use a color map to get a distinct color for each label
    colormap = plt.get_cmap("tab10", num_labels)
    colors = colormap(labels % num_labels)[:, :3]  # Get RGB values for labels (without alpha)

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()

    # Set the points in the point cloud
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Assign colors based on the labels
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name='Colored Point Cloud')

# ************************************************************************************

# Test round begin!
for test in range(6):
    do_mid_pp_projection(point_cloud, point_label, test, fov_value=1.6)

points_with_prediction = complete_labels_with_knn(k=5, point_cloud=point_cloud[:, 0:3])

for i in range(point_cloud.shape[0]):
    put = pack_coordinates(point_cloud[i, 0], point_cloud[i, 1], point_cloud[i, 2])
    gt_dataset.add_point_with_label(put, point_label[i])

bench_points = gt_dataset.most_voted_labels()

# get_3d_eval_res(np.array(points_with_prediction)[:, -1], np.array(bench_points)[:, -1])
#
# visualize_points_with_colored_labels_open3d(points_with_prediction)
# visualize_points_with_colored_labels_open3d(bench_points)