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

def do_perspective_projection(points_3d, label, target_type, fov_number, save_image, save_label, heights, widths, longitudes, ang, val):
    temp = np.zeros(points_3d.shape)
    if (target_type == "left"):
        # angle *= -1
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=ang, phi_y=0, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])
        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths * 1 / (val + 1), np.min(temp[:, 2]) + Y * heights / 2,
             np.min(temp[:, 0]) + Z * longitudes / 4])
        target = camera_position + (-0.1, 0, 0)
        reverse = True
    elif (target_type == "right"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=ang, phi_y=0, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / (val + 1), np.min(temp[:, 2]) + Y * heights / 2,
             np.min(temp[:, 0]) + Z * longitudes / 4])
        target = camera_position + (0.1, 0, 0)
        reverse = True
    elif (target_type == "forward"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=0, phi_y=ang, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / (val + 1), np.min(temp[:, 2]) + Y * heights / 2,
             np.min(temp[:, 0]) + Z * longitudes / 4])
        target = camera_position + (0, 0, 0.1)
        reverse = False
    elif (target_type == "back"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=0, phi_y=ang, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / (val + 1), np.min(temp[:, 2]) + Y * heights / 2,
             np.min(temp[:, 0]) + Z * longitudes / 4])
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
        original_indices = np.where(~mask)[0]
    else:
        valid_points = valid_points[mask]
        original_indices = np.where(mask)[0]

    valid_points = np.hstack((valid_points, original_indices[:, np.newaxis]))
    # valid_points = points
    # Transform the point through the whole pipeline
    new_coords = transform_point(np.transpose(valid_points), model_matrix, view_matrix, projection_matrix)

    # Project to screen coordinates
    screen_coords = project_to_screen(np.array(new_coords), width, height)

    x_coords = screen_coords[:, 0].astype(np.int32)

    y_coords = screen_coords[:, 1].astype(np.int32)

    colors = screen_coords[:, 3:6].astype(np.uint8)  # Assuming RGB values in uint8 format

    labels = screen_coords[:, 6:].astype(np.uint8)  # Assuming RGB values in uint8 format

    # index = screen_coords[:, -1].astype(np.uint8)

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
    proj_l[y_coords[valid_indices], x_coords[valid_indices]] = labels[valid_indices].reshape(-1)

    # if isinstance(image_rgb, np.ndarray):
    #     image_rgb = Image.fromarray(image_rgb)
    # else:
    #     image_rgb = image_rgb

    transform = get_transforms()
    augmented = transform(image=image_rgb, mask=proj_l)
    image_np, mask = augmented['image'], augmented['mask']

    # print("image_tensor before:", image_np.shape)
    # Convert numpy arrays to tensors and ensure correct dimensions
    image_tensor = torch.from_numpy(image_np.numpy()).float() / 255.0  # [H, W, C] -> [C, H, W]
    mask_tensor = torch.from_numpy(mask.numpy()).long()  # Ensure m

    test_dataset = [{'image': image_tensor, 'mask': mask_tensor}]
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                                                  drop_last=True)
    images, masks, preds = get_val_batch(test_dataloader, model)
    image_mask = masks[0].detach().cpu().float().reshape(masks[0].shape)
    label_image = preds[0].detach().cpu().float().reshape(preds[0].shape)

    label_image = postprocess_output(label_image)

    # num_colors = int(np.max(label_image) + 1)

    # Define a colormap with the specified number of colors
    # base_colormap = plt.get_cmap('viridis')
    # colormap = base_colormap(np.linspace(0, 1, num_colors))
    # norm = mcolors.Normalize(vmin=np.min(label_image), vmax=np.max(label_image))
    # colorized_image = mcolors.ListedColormap(colormap)(norm(label_image))

    # if not os.path.exists(base_dir + '/PP/Pred/'):
    #     # Create the folder if it doesn't exist
    #     os.makedirs(base_dir + '/PP/Pred/')

    # cv2.imwrite(base_dir + '/PP/Pred/' + 'label_image_pred' + '.png', colorized_image * 255)
    sub_pred = np.ones((valid_points.shape[0], valid_points.shape[1]-1))
    sub_pred[valid_indices, 0] = valid_points[valid_indices, 2]
    sub_pred[valid_indices, 1] = valid_points[valid_indices, 0]
    sub_pred[valid_indices, 2] = valid_points[valid_indices, 1]
    sub_pred[valid_indices, 3:6] = valid_points[valid_indices, 4:7]
    sub_pred[valid_indices, 7] = valid_points[valid_indices, -1]
    sub_pred[valid_indices, 6] = label_image[y_coords[valid_indices], x_coords[valid_indices]]
    # sub_pred[valid_indices, 6] = proj_l[y_coords[valid_indices], x_coords[valid_indices]]
    # return image_rgb, proj_l, sub_pred[:, 6:]
    return sub_pred[:, 6:]

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
    if label_image.shape[:2] != target_size:
        label_image = cv2.resize(label_image, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    return label_image

param_dict = {
    'auditorium': [2.2, 2, 1],
    'office': [1.5, 6, 1],
    'hallway': [1.2, 2, 1],
    'conferenceRoom': [1.8, 10, 1],
    'WC': [1.5, 6, 1],
    'pantry': [1.5, 6, 1],
    'storage': [1.2, 4, 1],
    'lounge': [1.5, 6, 1],
    'copyRoom': [1.5, 6, 1],
    'openspace': [1.5, 6, 1],
    'lobby': [1.2, 2, 1],
    'default': [1.5, 4, 1]
}

def gen_the_pp_image(scan, label, scan_path):
    pred_label = []
    k = [key for key in param_dict if key in scan_path.split(os.sep)[-3]]
    if len(k) > 0:
        k = k[0]
    else:
        k = 'default'
    fov = np.pi / param_dict[k][0]
    number = 0
    dataset = scan
    for h in np.arange(1):
        for w in np.arange(param_dict[k][2]):
            for l in np.arange(3):
                save_label_path = "/home/xi/repo/research_2/PP/label_test/left_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                  scan_path.split(os.sep)[-3] + '.label'
                save_image_path = "/home/xi/repo/research_2/PP/image_test/left_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                         scan_path.split(os.sep)[-3]
                pred_label.append(do_perspective_projection(points_3d=scan, label=label, target_type="left", save_image=save_image_path,
                                      save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number, val=param_dict[k][2]))

                # pred_label.append(la)

                save_label_path = "/home/xi/repo/research_2/PP/label_test/right_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3] + '.label'
                save_image_path = "/home/xi/repo/research_2/PP/image_test/right_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3]
                pred_label.append(do_perspective_projection(points_3d=scan, label=label, target_type="right", save_image=save_image_path,
                                          save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number, val=param_dict[k][2]))

                # pred_label.append(la)

                save_label_path = "/home/xi/repo/research_2/PP/label_test/forward_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3] + '.label'
                save_image_path = "/home/xi/repo/research_2/PP/image_test/forward_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3]
                pred_label.append(do_perspective_projection(points_3d=scan, label=label, target_type="forward", save_image=save_image_path,
                                          save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number, val=param_dict[k][2]))

                # pred_label.append(la)

                #
                save_label_path = "/home/xi/repo/research_2/PP/label_test/back_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3] + '.label'
                save_image_path = "/home/xi/repo/research_2/PP/image_test/back_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3]
                pred_label.append(do_perspective_projection(points_3d=dataset, label=label, target_type="back", save_image=save_image_path,
                                          save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number, val=param_dict[k][2]))

                # pred_label.append(la)

    return pred_label

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

from collections import Counter

class pj:
    def __init__(self, cloud):
        pj.cloud = -np.ones([cloud.shape[0], 7])
        pj.cloud[:, 0:6] = cloud
        pj.weight = {'-1': 0, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1, '11': 1, '12': 1, '13': 1}
        pj.priority = {'-1': 0, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1, '11': 1, '12': 1, '13': 1}
        pj.temp_label = [[-1 for _ in range(1)] for _ in range(pj.cloud.shape[0])]

    def predict_labels_weighted(self, labels):
        for sub in labels:
            for new_label, index in sub:  # Process each valid point and its label
                current_label = self.cloud[int(index), -1]  # Assume labels are stored in the last column
                current_priority = self.weight.get(str(int(current_label)), float('-inf'))
                new_priority = self.weight.get(str(int(new_label)), float('-inf'))

                # Update the label if the new one has higher priority
                if new_priority > current_priority:
                    self.cloud[int(index), -1] = new_label
        return self.cloud
    
    def update_most_voted_label(self):
        for i, l in enumerate(self.temp_label):
            new_label = Counter(l).most_common(1)[0][0] #Get the most voted label
            # Try to avoid the filled -1
            new_priority = self.priority.get(str(int(new_label)), float('-inf'))
            current_label = self.cloud[int(i), -1]  # Assume labels are stored in the last column
            current_priority = self.priority.get(str(int(current_label)), float('-inf'))

            # Update the label if the new one has higher priority
            if new_priority > current_priority:
                self.cloud[int(i), -1] = new_label

        return

    def predic_labels_appearance(self, labels):
        for sub in labels:
            for new_label, index in sub:  # Process each valid point and its label
                self.temp_label[index].append(new_label)

        self.update_most_voted_label()
        
        return self.cloud



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

from CNNSegmentation.CNN import *