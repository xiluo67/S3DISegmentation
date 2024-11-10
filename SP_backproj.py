import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import  torch
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image
import torch.nn as nn
import albumentations as A
import albumentations.pytorch as A_pytorch
import open3d as o3d
# -----------------------------------------------------
from CNN import *
"""
This file is used for spherical projection back projection process: projecting the predicted 2D 
image mask back to 3D space.
-----------------------------------------------------------------------------------------------------
Date: 2024.11

"""
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

def calculate_yaw(x, y, theta=0):

    x_prime = x * np.cos(theta) - y * np.sin(theta)
    y_prime = x * np.sin(theta) + y * np.cos(theta)

    return y_prime, x_prime

class SP_backproj():
    def __init__(self, points, CNN_model, gt_label):
        self.points = points
        self.model = CNN_model
        self.gt_label = gt_label

    def do_range_projection(self, proj_fov_up, proj_fov_down, save_image, save_label, proj_W, proj_H, label):
        """ Project a pointcloud into a spherical projection image.projection.
            Function takes no arguments because it can be also called externally
            if the value of the constructor was not set (in case you change your
            mind about wanting the projection)
        """
        # laser parameters
        fov_up = proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        x_means = np.mean(self.points[:, 0])
        y_means = np.mean(self.points[:, 1])
        z_means = np.mean(self.points[:, 2])
        # get scan components
        scan_x = self.points[:, 0] - x_means
        scan_y = self.points[:, 1] - y_means
        scan_z = self.points[:, 2] - z_means

        R = self.points[:, 3]
        G = self.points[:, 4]
        B = self.points[:, 5]

        # get depth of all points
        depth_points = np.zeros(self.points.shape)
        depth_points[:, 0] = scan_x - np.mean(scan_x)
        depth_points[:, 1] = scan_y - np.mean(scan_y)
        depth_points[:, 2] = scan_z - np.mean(scan_z)
        depth = np.linalg.norm(depth_points, 2, axis=1)
        # get angles of all points
        y_new, x_new = calculate_yaw(depth_points[:, 0], depth_points[:, 1], theta=0)
        yaw = -np.arctan2(y_new, x_new)
        # yaw = -np.arctan2(depth_points[:, 1], depth_points[:, 0])
        pitch = np.arcsin(depth_points[:, 2] / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= proj_W  # in [0.0, W]
        proj_y *= proj_H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(proj_W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(proj_H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]

        # copy of depth in original order
        unproj_range = np.copy(depth)

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        points = self.points[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        proj_range = np.zeros((proj_H, proj_W), dtype=np.int32)
        proj_xyz = np.zeros((proj_H, proj_W, 3), dtype=np.int32)
        proj_idx = np.zeros((proj_H, proj_W), dtype=np.float32)
        proj_r = np.zeros((proj_H, proj_W), dtype=np.float32)
        proj_g = np.zeros((proj_H, proj_W), dtype=np.float32)
        proj_b = np.zeros((proj_H, proj_W), dtype=np.float32)
        proj_l = np.zeros((proj_H, proj_W), dtype=np.int32)

        # assing to images
        proj_range[proj_y, proj_x] = depth
        proj_xyz[proj_y, proj_x] = points[:,0:3]
        proj_idx[proj_y, proj_x] = indices
        proj_r[proj_y, proj_x] = R[order]
        proj_g[proj_y, proj_x] = G[order]
        proj_b[proj_y, proj_x] = B[order]
        proj_l[proj_y, proj_x] = label[order]
        proj_mask = (proj_idx > 0).astype(np.int32)

        num_colors = np.max(proj_l) + 1
        #
        # # Define a colormap with the specified number of colors
        base_colormap = plt.get_cmap('viridis')
        colormap = base_colormap(np.linspace(0, 1, num_colors))
        norm = mcolors.Normalize(vmin=np.min(proj_l), vmax=np.max(proj_l))
        colorized_image = mcolors.ListedColormap(colormap)(norm(proj_l))

        #show image
        proj = np.zeros([proj_H, proj_W, 3], dtype=np.uint8)
        proj[:, :, 0] = proj_r
        proj[:, :, 1] = proj_g
        proj[:, :, 2] = proj_b
        # Save the projected mask into label folder
        if not os.path.exists("./SP/label/"):
            # Create the folder if it doesn't exist
            os.makedirs("./SP/label/")
        np.savetxt("./SP/label/" + save_label, proj_l)
        cv2.imwrite('./SP/label/label_image_gt_mask' + '.png', colorized_image*255)
        cv2.imwrite('./SP/label/label_image'+'.png', proj_l)

        # Save the projected image into image folder
        if not os.path.exists("./SP/image/"):
            # Create the folder if it doesn't exist
            os.makedirs("./SP/image/")
        plt.imshow(proj)
        plt.axis('off')
        plt.savefig("./SP/image/" + save_image + '.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        return proj

    def project_point_to_2d(self, label_image, proj_W, proj_H, proj_fov_up, proj_fov_down):
        # Laser parameters
        fov_up = proj_fov_up / 180.0 * np.pi  # field of view up in rad
        fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

        # Mean subtraction
        x_means = np.mean(self.points[:, 0])
        y_means = np.mean(self.points[:, 1])
        z_means = np.mean(self.points[:, 2])

        scan_x = self.points[:, 0] - x_means
        scan_y = self.points[:, 1] - y_means
        scan_z = self.points[:, 2] - z_means

        depth_points = np.zeros(self.points.shape)
        depth_points[:, 0] = scan_x - np.mean(scan_x)
        depth_points[:, 1] = scan_y - np.mean(scan_y)
        depth_points[:, 2] = scan_z - np.mean(scan_z)
        depth = np.linalg.norm(depth_points, 2, axis=1)

        # Calculate angles
        y_new, x_new = calculate_yaw(depth_points[:, 0], depth_points[:, 1])
        yaw = -np.arctan2(y_new, x_new)
        pitch = np.arcsin(depth_points[:, 2] / depth)

        # Project to image coordinates
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # Scale to image size
        proj_x *= proj_W  # in [0.0, W]
        proj_y *= proj_H  # in [0.0, H]

        # Determine pixel coordinates
        pixel_x = np.clip(np.round(proj_x).astype(int), 0, proj_W - 1)
        pixel_y = np.clip(np.round(proj_y).astype(int), 0, proj_H - 1)

        # Get labels for pixels
        pred_labels = np.full(pixel_x.shape, -1)

        # Get labels for pixels
        for i in range(len(pixel_x)):
            x = pixel_x[i]
            y = pixel_y[i]
            if 0 <= x < label_image.shape[1] and 0 <= y < label_image.shape[0]:
                pred_labels[i] = label_image[y, x]
            else:
                pred_labels[i] = -1

        return proj_x, proj_y, pred_labels

    def predict_labels_with_knn(self, label_image, proj_W, proj_H, proj_fov_up, proj_fov_down, k=5):
        # Project all points
        proj_x, proj_y, pred_labels = self.project_point_to_2d(label_image, proj_W, proj_H, proj_fov_up, proj_fov_down)
        # pred_labels = np.round(pred_labels).astype(int)
        # Extend points with projections and labels
        extended_points = np.hstack((self.points, proj_x.reshape(-1, 1), proj_y.reshape(-1, 1), pred_labels.reshape(-1, 1)))

        # Separate projected and unprojected points
        has_labels = pred_labels != -1  # Change this based on how missing labels are represented
        projected_points = extended_points[has_labels]
        unprojected_points = extended_points[~has_labels]

        # Extract features and labels
        projected_features = projected_points[:, :-3]  # Exclude proj_x, proj_y, pred_label
        projected_labels = projected_points[:, -1]  # Only pred_label
        unique_classes = np.unique(projected_labels)
        print("Unique classes in label:", unique_classes)

        if len(unprojected_points) > 0:
            # Apply KNN
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(projected_features[:,:3], projected_labels.astype(int))

            unprojected_features = unprojected_points[:, :-3]  # Exclude proj_x, proj_y, pred_label
            predicted_labels_for_unprojected = knn.predict(unprojected_features[:,:3])

            # Update labels for unprojected points
            unprojected_points[:, -1] = predicted_labels_for_unprojected

            # Combine results
            all_points = np.vstack((projected_points, unprojected_points))
        else:
            all_points = projected_points

        return all_points

transform = A.Compose([
        # A.HorizontalFlip(),
        # A.RandomRotate90(),
        # A.OneOf([
        #     A.RandomBrightnessContrast(),
        #     A.HueSaturationValue()
        # ], p=0.3),
        A.Resize(height=1024, width=1024, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A_pytorch.ToTensorV2()  # Ensure correct import and usage
    ], p=1.0)

def preprocess_image(image_array):
    if isinstance(image_array, Image.Image):
        # Convert PIL Image to NumPy array
        image_array = np.array(image_array)

        # Check if image_array is a valid NumPy array (3D: height, width, channels)
    if not isinstance(image_array, np.ndarray):
        raise TypeError("The input image must be a numpy array.")

        # Apply transformations: Pass the image as 'image=image'
    transformed = transform(image=image_array)  # Apply transform
    transformed_image = transformed['image']  # Get the transformed image

    # Add batch dimension for model input (C, H, W)
    transformed_image = transformed_image.unsqueeze(0)
    return transformed_image

def postprocess_output(output, target_size=(1024, 1024)):
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

def visualize_segmentation(original_image, segmented_image):
    # Display original and segmented images side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(segmented_image, cmap='jet')  # Use 'jet' colormap for multi-class segmentation
    ax[1].set_title('Segmented Image')
    ax[1].axis('off')

    plt.show()

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

# Get image
point_cloud = np.loadtxt("/home/xi/repo/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_2/room_data/auditorium_2.txt")
point_label = np.loadtxt("/home/xi/repo/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_2/room_data/auditorium_2.label")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------------------------------------------------
# model = DeepLabV3_Pretrained(num_classes=14).to(device)
model = UNet(num_classes=14).to(device)
if torch.cuda.device_count() >= 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)
    model = model.cuda()
# model = VGGSegmentation(num_classes).to(device)
# model = UNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load('/home/xi/repo/VGG/log/model_20241109_175124_.pth'))
model.eval()

bj = SP_backproj(point_cloud, model, gt_label=point_label)


images = bj.do_range_projection(save_image="sp", save_label="sp.label", proj_fov_up=110, proj_fov_down=-110, proj_W=1024,
                            proj_H=1024, label=point_label)


test_files = [f for f in os.listdir('./SP/label/') if f.endswith('.label')]
test_dataset = SegmentationDataset(image_folder='./SP/image/', mask_folder='./SP/label/',
                                        file_list=test_files, transform=get_transforms())
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=10, drop_last=True)
images, masks, preds = get_val_batch(test_dataloader, model)

#-----------------------------------------------------------------------------------------------------------------
# image_mask = cv2.imread("./SP/label/label_image.png", cv2.IMREAD_GRAYSCALE)
# image_mask = np.loadtxt("/home/xi/repo/research_2/SP/label/Area_1_0_conferenceRoom_1.label")
image_mask = masks[0].detach().cpu().float().reshape(masks[0].shape)

# Ensure label_image is on CPU and reshape to match image_mask's shape
label_image = preds[0].detach().cpu().float().reshape(preds[0].shape)

compute_2d_acc(label_image, image_mask)

label_image = postprocess_output(label_image)

num_colors = int(np.max(label_image) + 1)

# Define a colormap with the specified number of colors
base_colormap = plt.get_cmap('viridis')
colormap = base_colormap(np.linspace(0, 1, num_colors))
norm = mcolors.Normalize(vmin=np.min(label_image), vmax=np.max(label_image))
colorized_image = mcolors.ListedColormap(colormap)(norm(label_image))

if not os.path.exists("./SP/Pred/"):
    # Create the folder if it doesn't exist
    os.makedirs("./SP/Pred/")

cv2.imwrite('./SP/Pred/' + './label_image_pred'+'.png', colorized_image*255)
unique_classes = np.unique(label_image)
print("Unique classes in label prediction image:", unique_classes)


# extended_points_with_labels = bj.predict_labels_with_knn(label_image, proj_W=512, proj_H=512, proj_fov_up=110, proj_fov_down = -110)
extended_points_with_labels = bj.predict_labels_with_knn(image_mask, proj_W=1024, proj_H=1024, proj_fov_up=110, proj_fov_down = -110)
get_3d_eval_res(extended_points_with_labels[:, -1], point_label)
visualize_points_with_colored_labels_open3d(extended_points_with_labels)