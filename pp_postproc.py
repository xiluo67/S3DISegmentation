
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import open3d as o3d

class points_with_pred_labels(self):
    def __init__(self, points):
        # Assuming points is a list of 3D coordinates [(x1, y1, z1), (x2, y2, z2), ...]
        self.points = points
        self.labels = [[] for _ in range(len(points))]

        def add_label(self, index, label):
            # Add label to the point at the specified index
            if 0 <= index < len(self.points):
                self.labels[index].append(label)
            else:
                print(f"Index {index} out of range.")

        def get_labels(self, index):
            # Get all labels for the point at the specified index
            if 0 <= index < len(self.points):
                return self.labels[index]
            else:
                print(f"Index {index} out of range.")
                return None

        def most_voted_labels(self):
            # Initialize an array to hold the (x, y, z, most_voted_label) for each point
            points_with_labels = []

            for point, label_list in zip(self.points[:, 0:3], self.labels):
                if not label_list:
                    # If there are no labels, append None for the label
                    points_with_labels.append((*point, -1))
                    continue

                # Count the occurrences of each label
                label_counts = {}
                for label in label_list:
                    label_counts[label] = label_counts.get(label, 0) + 1

                # Find the label(s) with the maximum votes
                max_count = max(label_counts.values())
                max_labels = [label for label, count in label_counts.items() if count == max_count]

                # Determine the most voted label or set to None if there's a tie
                if len(max_labels) > 1:
                    points_with_labels.append((*point, -1))
                else:
                    points_with_labels.append((*point, max_labels[0]))

            return points_with_labels

def complete_labels_with_knn(k=5):
    # Project all points
    extended_points = point_cloud_dataset.most_voted_labels()

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
    "/media/rosie/KINGSTON/Dataset_2/Stanford3dDataset_v1.2_Aligned_Version/Area_4/conferenceRoom_2/room_data/conferenceRoom_2.txt")
point_label = np.loadtxt(
    "/media/rosie/KINGSTON/Dataset_2/Stanford3dDataset_v1.2_Aligned_Version/Area_4/conferenceRoom_2/room_data/conferenceRoom_2.label")
model = UNet(num_classes=num_classes).to(device)
if torch.cuda.device_count() >= 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)
    model = model.cuda()
model.load_state_dict(torch.load('/home/xi/repo/VGG/log/512_SP_UNet.pth'))
model.eval()

point_cloud_dataset = points_with_pred_labels(point_cloud)
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
    projection_matrix = perspective_projection(fov_value, aspect_ratio, near, far)
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
    screen_coords = project_to_screen(np.array(new_coords), width, height)

    x_coords = screen_coords[:, 0].astype(np.int32)

    y_coords = screen_coords[:, 1].astype(np.int32)

    colors = screen_coords[:, 3:6].astype(np.uint8)  # Assuming RGB values in uint8 format

    labels = screen_coords[:, 6:].astype(np.uint8)  # Assuming RGB values in uint8 format

    # Calculate the actual dimensions of the image
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
        label_pred_2d = postprocess_output(label_image)

    point_cloud_dataset.add_label(valid_indices, label_pred_2d[y_coords[valid_indices], x_coords[valid_indices]])


def get_3d_eval_res(predicted_labels, ground_truth_labels):
    # Determine the unique classes from the ground truth labels
    classes = np.unique(ground_truth_labels)
    num_classes = len(classes)

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

points_with_prediction = complete_labels_with_knn(k=5)

get_3d_eval_res(points_with_prediction[:, -1], point_label)

visualize_points_with_colored_labels_open3d(points_with_prediction)