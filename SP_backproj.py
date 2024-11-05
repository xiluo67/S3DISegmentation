import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import  torch

"""
This file is used for spherical projection back projection process: projecting the predicted 2D 
image mask back to 3D space.
-----------------------------------------------------------------------------------------------------
Date: 2024.11

"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def calculate_yaw(x, y, theta=0):

    x_prime = x * np.cos(theta) - y * np.sin(theta)
    y_prime = x * np.sin(theta) + y * np.cos(theta)

    return y_prime, x_prime

def do_range_projection(points, proj_fov_up, proj_fov_down, save_image, save_label, proj_W, proj_H, R, G, B, label):
    """ Project a pointcloud into a spherical projection image.projection.
        Function takes no arguments because it can be also called externally
        if the value of the constructor was not set (in case you change your
        mind about wanting the projection)
    """
    # laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi  # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

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
    points = points[order]
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

    # Define a colormap with the specified number of colors
    base_colormap = plt.colormaps.get_cmap('viridis')
    colormap = base_colormap(np.linspace(0, 1, num_colors))
    norm = mcolors.Normalize(vmin=np.min(proj_l), vmax=np.max(proj_l))
    colorized_image = mcolors.ListedColormap(colormap)(norm(proj_l))

    #show image
    proj = np.zeros([proj_H, proj_W, 3], dtype=np.uint8)
    proj[:, :, 0] = proj_r
    proj[:, :, 1] = proj_g
    proj[:, :, 2] = proj_b

    np.savetxt(save_label, proj_l)

    plt.imshow(colorized_image)
    plt.axis('off')
    # plt.savefig(save_name+'.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    # plt.imshow(proj)
    # plt.axis('off')
    # plt.savefig( save_image+'.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    image_rgb = cv2.cvtColor(proj, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_image+'.png', image_rgb)
    return image_rgb


def project_point_to_2d(points, label_image, proj_W, proj_H, proj_fov_up, proj_fov_down):
    # Laser parameters
    fov_up = proj_fov_up / 180.0 * np.pi  # field of view up in rad
    fov_down = proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # Mean subtraction
    x_means = np.mean(points[:, 0])
    y_means = np.mean(points[:, 1])
    z_means = np.mean(points[:, 2])

    scan_x = points[:, 0] - x_means
    scan_y = points[:, 1] - y_means
    scan_z = points[:, 2] - z_means

    depth_points = np.zeros(points.shape)
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

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def predict_labels_with_knn(point_cloud, label_image, proj_W, proj_H, proj_fov_up, proj_fov_down, k=5):
    # Project all points
    proj_x, proj_y, pred_labels = project_point_to_2d(point_cloud, label_image, proj_W, proj_H, proj_fov_up, proj_fov_down)
    # pred_labels = np.round(pred_labels).astype(int)
    # Extend points with projections and labels
    extended_points = np.hstack((point_cloud, proj_x.reshape(-1, 1), proj_y.reshape(-1, 1), pred_labels.reshape(-1, 1)))

    # Separate projected and unprojected points
    has_labels = pred_labels != -1  # Change this based on how missing labels are represented
    projected_points = extended_points[has_labels]
    unprojected_points = extended_points[~has_labels]

    # Extract features and labels
    projected_features = projected_points[:, :-3]  # Exclude proj_x, proj_y, pred_label
    projected_labels = projected_points[:, -1]  # Only pred_label
    unique_classes = np.unique(projected_labels)
    print("Unique classes:", unique_classes)

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

from testing_group_2 import *
from testing_group_3 import *
# from get_image import do_range_projection

# Get image
point_cloud = np.loadtxt("/media/rosie/KINGSTON/Dataset_2/Stanford3dDataset_v1.2_Aligned_Version/Area_4/conferenceRoom_2/room_data/conferenceRoom_2.txt")
point_label = np.loadtxt("/media/rosie/KINGSTON/Dataset_2/Stanford3dDataset_v1.2_Aligned_Version/Area_4/conferenceRoom_2/room_data/conferenceRoom_2.label")
images = do_range_projection(point_cloud, save_image="/home/rosie/Concordia/research/pred/sp_image", save_label="/home/rosie/Concordia/research/pred/sp_gt.label", proj_fov_up=110, proj_fov_down=-110, proj_W=512,
                            proj_H=512, R=point_cloud[:, 3], G=point_cloud[:, 4], B=point_cloud[:, 5], label = point_label)
# -------------------------------------------------------------------
# model = DeepLabV3_Pretrained(num_classes=14).to(device)
model = SegFormerPretrained(num_classes=14).to(device)
model.load_state_dict(torch.load('./log/Dataset1_SP_SegF.pth'))
model.eval()
#-----------------------------------------------------------------------------------------------------------------
# Load and preprocess the image
from PIL import Image
def preprocess_image(image_array):
    # Convert to PIL Image if necessary
    if isinstance(image_array, np.ndarray):
        image = Image.fromarray(image_array)
    else:
        image = image_array

    # Define the preprocessing transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # Adjust to match your model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
    ])

    # Apply transformations
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


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

import open3d as o3d

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

# Example usage
image_array = images  # Load your image array
preprocessed_image = preprocess_image(image_array)

# Perform segmentation
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
    unique_classes = np.unique(label_image)
    print("Unique classes in label image:", unique_classes)

extended_points_with_labels = predict_labels_with_knn(point_cloud, label_image, proj_W=512, proj_H=512, proj_fov_up=110, proj_fov_down = -110)

visualize_points_with_colored_labels_open3d(extended_points_with_labels)