import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

"""
This file is used for perspective projection back projection process: projecting the predicted 2D 
image mask back to 3D space.
-----------------------------------------------------------------------------------------------------
Date: 2024.11

"""

# from sympy.strategies.core import switch
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


def rotate_3d(origin_points, theta_x, phi_y, psi_z):
    points = origin_points
    theta_x = np.radians(theta_x)
    phi_y = np.radians(phi_y)
    psi_z = np.radians(psi_z)

    # Rotation matrix for x-axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

    # Rotation matrix for y-axis
    R_y = np.array([[np.cos(phi_y), 0, np.sin(phi_y)],
                    [0, 1, 0],
                    [-np.sin(phi_y), 0, np.cos(phi_y)]])

    # Rotation matrix for z-axis
    R_z = np.array([[np.cos(psi_z), -np.sin(psi_z), 0],
                    [np.sin(psi_z), np.cos(psi_z), 0],
                    [0, 0, 1]])

    # Combine the rotations by multiplying them in the order: Rz * Ry * Rx
    rotation_matrix = R_z @ R_y @ R_x

    # Apply the rotation matrix to the entire point cloud
    rotated_points = points @ rotation_matrix.T  # Apply to all points at once

    return rotated_points

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


def transform_point(point, model_matrix, view_matrix, projection_matrix):
    world_coords = np.dot(model_matrix, point[0:4, :])
    camera_coords = np.dot(view_matrix, world_coords)
    clip_coords = np.dot(projection_matrix, camera_coords)
    # ndc_coords = clip_coords[:3, :] / clip_coords[3, :]  # Perspective division
    ndc_coords = np.vstack(
        ((clip_coords[:3, :] / clip_coords[3, :]), point[4, :], point[5, :], point[6, :], point[7, :]))
    return np.transpose(ndc_coords)

# Pre-define the view params (location, view angle... ,etc.)
width, height = 512, 512
# fov = np.pi / 1.4  # 60 degrees
near, far = 0.1, 1000
aspect_ratio = width / height

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(num_classes=14).to(device)
if torch.cuda.device_count() >= 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)
    model = model.cuda()
model.load_state_dict(torch.load('/home/xi/repo/VGG/log//model_20241015_120101_.pth'))
model.eval()


from sklearn.neighbors import KNeighborsClassifier
whole_set = []
def do_perspective_projection(points_3d, label, target_type, fov_number, save_image, save_label, heights, widths, longitudes, ang):

    temp = np.zeros(points_3d.shape)
    if (target_type == "left"):
        # angle *= -1
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=ang, phi_y=0, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]


        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / 9, np.min(temp[:, 2]) + Y * heights / 9, np.min(temp[:, 0]) + Z * longitudes / 9])
        target = camera_position + (-0.1, 0, 0)
        reverse = True
    elif (target_type == "right"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=ang, phi_y=0, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / 9, np.min(temp[:, 2]) + Y * heights / 9, np.min(temp[:, 0]) + Z * longitudes / 9])
        target = camera_position + (0.1, 0, 0)
        reverse = True
    elif (target_type == "forward"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=0, phi_y=ang, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / 9, np.min(temp[:, 2]) + Y * heights / 9, np.min(temp[:, 0]) + Z * longitudes / 9])
        target = camera_position + (0, 0, 0.1)
        reverse = False
    elif (target_type == "back"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=0, phi_y=ang, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / 9, np.min(temp[:, 2]) + Y * heights / 9, np.min(temp[:, 0]) + Z * longitudes / 9])
        target = camera_position + (0, 0, -0.1)
        reverse = False

    # Define model matrix (identity for simplicity)
    model_matrix = np.identity(4)
    up = np.array([0, 1, 0])
    # Compute view and projection matrices
    view_matrix = look_at(camera_position, target, up)
    projection_matrix = perspective_projection(fov_number, aspect_ratio, near, far)
    # Define a 3D point in homogeneous coordinates
    valid_points = np.hstack((temp[:, 1].reshape(-1, 1), temp[:, 2].reshape(-1, 1), temp[:, 0].reshape(-1, 1),
                        np.ones((temp.shape[0], 1)), temp[:, 3:], label.reshape(-1, 1)))  # (x, y, z, w)

    # filter out the 3d points behind the camera and looking direction
    forward = np.float_(target - camera_position)
    forward /= np.linalg.norm(forward)
    dot_products = np.dot((valid_points[:, 0:3] - camera_position), forward)
    mask = dot_products > 0
    if (reverse):
        valid_points = valid_points[~mask]
    else:
        valid_points = valid_points[mask]

    # valid_points = points
    # Transform the point through the whole pipeline
    new_coords = transform_point(np.transpose(valid_points), model_matrix, view_matrix, projection_matrix)

    # Project to screen coordinates
    screen_coords = project_to_screen(np.array(new_coords), width, height)

    x_coords = screen_coords[:, 0].astype(np.int32)

    y_coords = screen_coords[:, 1].astype(np.int32)

    colors = screen_coords[:, 3:6].astype(np.uint8)  # Assuming RGB values in uint8 format

    labels = screen_coords[:, 6:].astype(np.uint8)  # Assuming RGB values in uint8 format

    # Calculate the actual dimensions of the image
    img_width = width
    img_height = height

    # Create a blank image with specified dimensions
    image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    proj_l = np.zeros((img_height, img_width), dtype=np.uint8)
    # Assign colors to corresponding coordinates within the limits
    valid_indices = (x_coords < img_width) & (y_coords < img_height) & (0 <= x_coords) & (0 <= y_coords)
    image[y_coords[valid_indices], x_coords[valid_indices]] = colors[valid_indices]

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

    proj_l[y_coords[valid_indices], x_coords[valid_indices]] = labels[valid_indices].reshape(-1)
    sub_pred = np.ones((valid_points.shape[0], valid_points.shape[1]-1))
    sub_pred[valid_indices, 0] = valid_points[valid_indices, 2]
    sub_pred[valid_indices, 1] = valid_points[valid_indices, 0]
    sub_pred[valid_indices, 2] = valid_points[valid_indices, 1]
    sub_pred[valid_indices, 3:6] = valid_points[valid_indices, 4:7]

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
    label_image = torch.argmax(logits, dim=1).squeeze(0)  # Shape: [batch_size, height, width]
    # sub_pred[valid_indices, 6] = label_image.cpu()[y_coords[valid_indices], x_coords[valid_indices]]
    sub_pred[valid_indices, 6] = proj_l[y_coords[valid_indices], x_coords[valid_indices]]
    whole_set.append(np.transpose(sub_pred))
def gen_the_pp_image(scan, label, scan_path):
    fov = np.pi / 1.5
    for f in np.arange(3):
        number = f*6 - 6
        # number = 0
        dataset = scan
        for h in np.arange(7):
            for w in np.arange(7):
                for l in np.arange(7):
                    save_label_path = "/home/xi/repo/research_2/PP/label_test/left_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                      scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image_test/left_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="left", save_image=save_image_path,
                                          save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number)

                    save_label_path = "/home/xi/repo/research_2/PP/label_test/right_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image_test/right_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="right", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number)

                    save_label_path = "/home/xi/repo/research_2/PP/label_test/forward_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image_test/forward_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="forward", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number)
                    #
                    save_label_path = "/home/xi/repo/research_2/PP/label_test/back_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image_test/back_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=dataset, label=label, target_type="back", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number)

# base_dir = '/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/'
# base_dir = '/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_1/'
base_dir = '/home/xi/repo/new/'
def find_txt_files(base_dir):
    # List to store paths to all .txt files
    txt_files = []
    label_files = []

    # Walk through the base directory
    for root, dirs, files in os.walk(base_dir):
        # Check if 'room_data' is in the path
        if 'room_data' in root:
            # Iterate over files in the current 'room_data' folder
            for file in files:
                if file.endswith('.txt'):
                    # Append the full path of the .txt file
                    txt_files.append(os.path.join(root, file))
                elif file.endswith('.label'):
                    # Append the full path of the .label file
                    label_files.append(os.path.join(root, file))

    return txt_files, label_files
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
    # Calculate overall accuracy
    overall_accuracy = global_tp / (global_gt_total) if global_gt_total > 0 else 0.0

    # Calculate overall IoU
    overall_iou = np.sum(iou) / num_classes if num_classes > 0 else 0.0

    # Print overall metrics
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall IoU: {overall_iou:.4f}")


scan_files, scan_labels = find_txt_files(base_dir)

# Print all .txt file paths
for txt_file in scan_files:
    print(txt_file)

for i in np.arange(len(scan_files)):
    scan_path = scan_files[i]
    label_path = scan_labels[i]

    scan = np.loadtxt(scan_path, dtype=np.float32)
    # scan = scan.reshape((-1, 6))
    label = np.loadtxt(label_path, dtype=np.float32)
    gen_the_pp_image(scan=scan, label=label, scan_path=scan_path)

    stacked_array = np.hstack(whole_set)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(stacked_array[:6, :].T, stacked_array[6, :])
    pre_l = knn.predict(scan)
    get_3d_eval_res(pre_l, label)

