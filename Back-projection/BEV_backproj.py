import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import matplotlib.colors as mcolors
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import open3d as o3d


import matplotlib.pyplot as plt
def visualize_points_with_colored_labels_open3d_fix(points):
    # Assuming points is an (N, 9) array where:
    # points[:, 0:3] -> xyz coordinates
    # points[:, -1]   -> labels (l)

    xyz = points[:, 0:3]  # Extract xyz coordinates
    labels = points[:, -1].astype(int)  # Extract labels and convert to integers

    # Define a fixed colormap for labels 0-13
    fixed_colormap = [
        [0.121, 0.466, 0.705],  # Label 0
        [1.000, 0.498, 0.054],  # Label 1
        [0.172, 0.627, 0.172],  # Label 2
        [0.839, 0.153, 0.157],  # Label 3
        [0.580, 0.403, 0.741],  # Label 4
        [0.549, 0.337, 0.294],  # Label 5
        [0.890, 0.467, 0.761],  # Label 6
        [0.498, 0.498, 0.498],  # Label 7
        [0.737, 0.741, 0.133],  # Label 8
        [0.090, 0.745, 0.812],  # Label 9
        [0.9, 0.6, 0.0],        # Label 10
        [0.5, 0.0, 0.5],        # Label 11
        [0.0, 0.5, 0.5],        # Label 12
        [0.5, 0.5, 0.0],        # Label 13
        [0.8, 0.5, 0.0],        # Label 14
        [0.7, 0.7, 0.7]
    ]

    # Handle out-of-range labels (in case you have labels beyond 0-13)
    colors = np.zeros((len(labels), 3))
    for i, label in enumerate(labels):
        if 0 <= label < len(fixed_colormap):
            colors[i] = fixed_colormap[label]
        else:
            # Assign black or some fallback color for unexpected labels
            colors[i] = [0, 0, 0]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name='Fixed Color Point Cloud')
class pj:
    def __init__(self, cloud):
        pj.cloud = -np.ones([cloud.shape[0], 7])
        pj.cloud[:, 0:6] = cloud
        # pj.weight = {'-1': 0, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1, '11': 1, '12': 1, '13': 1}
        pj.priority = {'-1': 0, '0': 1, '1': 1, '2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 1, '8': 1, '9': 1, '10': 1,
                       '11': 1, '12': 1, '13': 1}
        pj.weight = {'-1': 0, '0': 0.0027, '1': 0.8357, '2': 0.0135, '3': 0.0008, '4': 0.0012, '5': 0.0014, '6': 0.0023,
                     '7': 0.0079, '8': 0.0128, '9': 0.0201,
                     '10': 0.0884, '11': 0.0068, '12': 0.0064}
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
            new_label = Counter(l).most_common(1)[0][0]  # Get the most voted label
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
                self.temp_label[int(index)].append(int(new_label))

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

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

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

def point_cloud_2_birdseye(save_mask, height_range, points, R, G, B, label, proj_W=512, proj_H=512,
                           res=0.005):
    """ Creates an 2D birds eye view representation of the point cloud data.
    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS

    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]

    # for unproj_range
    x_means = np.mean(points[:, 0])
    y_means = np.mean(points[:, 1])
    z_means = np.mean(points[:, 2])
    # get scan components
    scan_x = points[:, 0] - x_means
    scan_y = points[:, 1] - y_means
    scan_z = points[:, 2] - z_means

    # get depth of all points
    depth_points = np.zeros(points.shape)
    depth_points[:, 0] = scan_x - np.mean(scan_x)
    depth_points[:, 1] = scan_y - np.mean(scan_y)
    depth_points[:, 2] = scan_z - np.mean(scan_z)

    # BEV Projection
    fwd_range = (np.min(x_points), np.max(x_points))
    side_range = (np.min(y_points), np.max(y_points))


    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    v_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filter = np.logical_and(f_filt, v_filt)
    indices = np.argwhere(filter).flatten()
    valid_points = points[indices]
    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    R = R[indices]
    G = G[indices]
    B = B[indices]
    label = label[indices]
    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(side_range[0] / res)
    y_img -= int(fwd_range[0] / res)
    # y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    height_values = scale_to_255(pixel_values,
                                 min=np.min(z_points),
                                 max=np.max(z_points))

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = int(np.max(y_img)) + 1
    y_max = int(np.max(x_img)) + 1
    proj_r = np.zeros([proj_H, proj_W], dtype=np.uint8)
    proj_g = -10 * np.zeros([proj_H, proj_W], dtype=np.uint8)
    proj_b = -10 * np.zeros([proj_H, proj_W], dtype=np.uint8)
    proj_l = -10 * np.zeros([proj_H, proj_W], dtype=np.uint8)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    ia = indices.shape[0]
    for i in np.arange(ia):
        y = int((y_img[i] / x_max) * proj_W)
        x = int((x_img[i] / y_max) * proj_H)
        proj_r[x, y] = R[i]
        proj_g[x, y] = G[i]
        proj_b[x, y] = B[i]
        proj_l[x, y] = label[i]

    # show image
    proj = np.zeros([proj_H, proj_W, 3], dtype=np.uint8)
    proj[:, :, 0] = proj_r
    proj[:, :, 1] = proj_g
    proj[:, :, 2] = proj_b

    transform = get_transforms()
    augmented = transform(image=proj, mask=proj_l)
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
    # label_image = image_mask

    label_image = postprocess_output(label_image)

    num_classes = 15
    color_list = plt.get_cmap('tab20')(np.arange(num_classes))
    fixed_colormap = mcolors.ListedColormap(color_list)

    # Plot the first image
    plt.imshow(image_mask, cmap=fixed_colormap, vmin=0, vmax=num_classes - 1)
    plt.axis('off')
    plt.savefig(save_mask + '_gt.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot the second image
    plt.imshow(label_image, cmap=fixed_colormap, vmin=0, vmax=num_classes - 1)
    plt.axis('off')
    plt.savefig(save_mask + '_pred.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.imshow(proj)
    # plt.axis('off')
    # plt.savefig(save_image + '.jpg', bbox_inches='tight', pad_inches=0)
    # plt.close()
    # np.savetxt(save_label, proj_l)
    sub_pred = np.ones((valid_points.shape[0], 2))
    for poi in range(valid_points.shape[0]):
        sub_pred[poi, 1] = indices[poi]
        sub_pred[poi, 0] = label_image[int((x_img[poi] / y_max) * proj_H), int((y_img[poi] / x_max) * proj_W)]
    return sub_pred

from CNNSegmentation.CNN import *
num_classes = 15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Segformer.to(device)
# model = UNet(num_classes=14).to(device)
model = get_pretrianed_unet().to(device)
# model = SegFormerPretrained(num_classes=num_classes)
if torch.cuda.device_count() >= 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)
    model = model.cuda()
model.load_state_dict(torch.load('/home/xi/repo/VGG/log/model_UNET16_BEV3_TL_LR1.2-e03.pth'))
model.eval()


IoU_res = []
# ------------------------------------------------------------------------------------
# base_dir = '/home/xi/repo/Stanford3dDataset_v1.2_Aligned_Version_Test/'
base_dir = '/home/xi/repo/3sdis/Area_5'
# base_dir = '/home/xi/repo/new/test'
# base_dir = '/home/xi/repo/new2'
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

    pred_label= []
    hli = (np.max(scan[:, 2]) - np.min(scan[:, 2])) / 16
    for j in range(16):
        start = hli * j
        end = hli * (j + 1)
        height_range = (np.min(scan[:, 2]) + start - hli / 2, np.min(scan[:, 2]) + end + hli / 2)
        save_name = str(j) + "_" + os.path.basename(scan_path).replace('.txt', '')
        pred_label.append(point_cloud_2_birdseye(save_mask='/home/xi/Pictures/2D_mask/BEV/' + save_name, height_range=height_range, points=scan[:, 0:3], R=scan[:, 3], G=scan[:, 4], B=scan[:, 5], label=label))

    bj = pj(scan)
    # extended_points_with_labels = bj.predict_labels_weighted(l)
    extended_points_with_labels = bj.predic_labels_appearance(pred_label)
    _, _, _, IoU = get_3d_eval_res(extended_points_with_labels[:, -1], label)
    new_point = np.zeros((extended_points_with_labels.shape))
    new_point[:, 0:6] = scan
    new_point[:, 6] = label
    # visualize_points_with_colored_labels_open3d_fix(new_point)
    # visualize_points_with_colored_labels_open3d_fix(extended_points_with_labels)

    IoU_res.append(IoU)


print(np.mean(IoU_res))
