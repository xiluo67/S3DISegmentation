import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import open3d as o3d
# from sympy.strategies.core import switch

"""
This file is used for Perspective Projection: projecting 3D points to 2D plane (Optimized Version, adjust
parameters like fov/camera position) for different room type
----------------------------------------------------------------------------------------------------------
Date: 2024.11

"""

def vis(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color/255)
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], window_name="mesh pcd")
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
param_dict = {
    'auditorium': [2.2, 2, 3],
    'office': [1.5, 6, 3],
    'hallway': [1.2, 2, 1],
    'conferenceRoom': [1.8, 10, 3],
    'WC': [1.5, 6, 1],
    'pantry': [1.5, 6, 3],
    'storage': [1.2, 4, 1],
    'lounge': [1.5, 6, 3],
    'copyRoom': [1.5, 6, 3],
    'openspace': [1.5, 6, 3],
    'lobby': [1.2, 2, 1],
    'default': [1.5, 4, 3]
}
from scipy.ndimage import measurements
# why = np.loadtxt("/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_1/room_data/auditorium_1.txt")
# Function to fill voids with nearest neighbor interpolation
def fill_voids(image):
    # Create a copy of the image
    filled_image = np.copy(image)

    # Create a mask of the original non-zero pixels
    mask = filled_image > 0

    # Label the connected components in the mask
    labeled_array, num_features = measurements.label(mask)

    # Iterate over each labeled region and fill voids
    for label in range(1, num_features + 1):
        # Get the indices of the current region
        region_indices = np.argwhere(labeled_array == label)

        # Get the values of the current region
        region_values = filled_image[region_indices[:, 0], region_indices[:, 1]]

        # Calculate the mean of the current region values
        mean_value = np.mean(region_values)

        # Fill voids in the region with the mean value
        for idx in region_indices:
            if filled_image[idx[0], idx[1]] == 0:  # Only fill voids
                filled_image[idx[0], idx[1]] = mean_value

    return filled_image
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
            [np.min(temp[:, 1]) + X * widths * 1 / (val+1), np.min(temp[:, 2]) + Y * heights / 4, np.min(temp[:, 0]) + Z * longitudes / 4])
        target = camera_position + (-0.1, 0, 0)
        reverse = True
    elif (target_type == "right"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=ang, phi_y=0, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / (val+1), np.min(temp[:, 2]) + Y * heights / 4, np.min(temp[:, 0]) + Z * longitudes / 4])
        target = camera_position + (0.1, 0, 0)
        reverse = True
    elif (target_type == "forward"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=0, phi_y=ang, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / (val+1), np.min(temp[:, 2]) + Y * heights / 4, np.min(temp[:, 0]) + Z * longitudes / 4])
        target = camera_position + (0, 0, 0.1)
        reverse = False
    elif (target_type == "back"):
        temp[:, 0:3] = rotate_3d(points_3d[:, 0:3], theta_x=0, phi_y=ang, psi_z=0)
        temp[:, 3:] = points_3d[:, 3:]

        X = np.max(temp[:, 1]) - np.min(temp[:, 1])
        Y = np.max(temp[:, 2]) - np.min(temp[:, 2])
        Z = np.max(temp[:, 0]) - np.min(temp[:, 0])

        camera_position = np.array(
            [np.min(temp[:, 1]) + X * widths / (val+1), np.min(temp[:, 2]) + Y * heights / 4, np.min(temp[:, 0]) + Z * longitudes / 4])
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

    proj_l[y_coords[valid_indices], x_coords[valid_indices]] = labels[valid_indices].reshape(-1)
    if (len(np.unique(proj_l)) >= 0):
        # print(np.unique(proj_l))
        np.savetxt(save_label, proj_l)
        cv2.imwrite(save_image+'.jpg', fill_voids(image_rgb))

def gen_the_pp_image(scan, label, scan_path):
    k = [key for key in param_dict if key in scan_path.split(os.sep)[-3]]
    if len(k) > 0:
        k = k[0]
    else:
        k = 'default'
    fov = np.pi / param_dict[k][0]
    for f in np.arange(3):
        number = f * (param_dict[k][1]) - param_dict[k][1]
        dataset = scan
        for h in np.arange(3):
            for w in np.arange(param_dict[k][2]):
                for l in np.arange(3):
                    save_label_path = "/home/xi/repo/research_2/PP/label_f/left_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                      scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image_f/left_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="left", save_image=save_image_path,
                                          save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number, val=param_dict[k][2])

                    save_label_path = "/home/xi/repo/research_2/PP/label_f/right_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image_f/right_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="right", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number, val=param_dict[k][2])

                    save_label_path = "/home/xi/repo/research_2/PP/label_f/forward_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image_f/forward_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="forward", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number, val=param_dict[k][2])
                    #
                    save_label_path = "/home/xi/repo/research_2/PP/label_f/back_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image_f/back_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=dataset, label=label, target_type="back", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number, val=param_dict[k][2])

# base_dir = '/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_2'
# base_dir = '/home/xi/repo/Stanford3dDataset_v1.2_Aligned_Version_Test/'
# base_dir = '/home/xi/repo/Stanford3dDataset_v1.2_Aligned_Version_Test'
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


