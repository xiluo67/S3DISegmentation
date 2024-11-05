import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
# from sympy.strategies.core import switch

"""
This file is used for Perspective Projection: projecting 3D points to 2D plane
----------------------------------------------------------------------------------------------------------
Date: 2024.11

"""
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
    rotated_points = origin_points @ rotation_matrix.T  # Apply to all points at once

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

# why = np.loadtxt("/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_1/room_data/auditorium_1.txt")

def do_perspective_projection(points_3d, label, target_type, fov_number, save_image, save_label, heights, widths, longitudes, ang):
    points_3d = np.loadtxt("/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_1/room_data/auditorium_1.txt")
    if (target_type == "left"):
        # angle *= -1
        points_data = points_3d
        x = points_data[:, 0:3]
        points_data[:, 0:3] = rotate_3d(x, theta_x=ang, phi_y=0, psi_z=0)

        X = np.max(points_data[:, 1]) - np.min(points_data[:, 1])
        Y = np.max(points_data[:, 2]) - np.min(points_data[:, 2])
        Z = np.max(points_data[:, 0]) - np.min(points_data[:, 0])

        camera_position = np.array(
            [np.min(points_data[:, 1]) + X * 1 / 4, np.min(points_data[:, 2]) + Y * 1 / 4, np.min(points_data[:, 0]) + Z * 1 / 4])
        target = camera_position + (-0.1, 0, 0)
        reverse = True
    elif (target_type == "right"):
        points_data = points_3d
        x = points_data[:, 0:3]
        points_data[:, 0:3] = rotate_3d(x, theta_x=ang, phi_y=0, psi_z=0)

        X = np.max(points_data[:, 1]) - np.min(points_data[:, 1])
        Y = np.max(points_data[:, 2]) - np.min(points_data[:, 2])
        Z = np.max(points_data[:, 0]) - np.min(points_data[:, 0])

        camera_position = np.array(
            [np.min(points_data[:, 1]) + X / 4, np.min(points_data[:, 2]) + Y / 4, np.min(points_data[:, 0]) + Z / 4])
        target = camera_position + (0.1, 0, 0)
        reverse = True
    elif (target_type == "forward"):
        points_data = points_3d
        x = points_data[:, 0:3]
        points_data[:, 0:3] = rotate_3d(x, theta_x=0, phi_y=ang, psi_z=0)

        X = np.max(points_data[:, 1]) - np.min(points_data[:, 1])
        Y = np.max(points_data[:, 2]) - np.min(points_data[:, 2])
        Z = np.max(points_data[:, 0]) - np.min(points_data[:, 0])

        camera_position = np.array(
            [np.min(points_data[:, 1]) + X / 4, np.min(points_data[:, 2]) + Y / 4, np.min(points_data[:, 0]) + Z / 4])
        target = camera_position + (0, 0, 0.1)
        reverse = False
    elif (target_type == "back"):
        points_data = points_3d
        x = points_data[:, 0:3]
        # angle *= -1
        points_data[:, 0:3] = rotate_3d(x, theta_x=0, phi_y=ang, psi_z=0)

        X = np.max(points_data[:, 1]) - np.min(points_data[:, 1])
        Y = np.max(points_data[:, 2]) - np.min(points_data[:, 2])
        Z = np.max(points_data[:, 0]) - np.min(points_data[:, 0])

        camera_position = np.array(
            [np.min(points_data[:, 1]) + X / 4, np.min(points_data[:, 2]) + Y / 4, np.min(points_data[:, 0]) + Z / 4])
        target = camera_position + (0, 0, -0.1)
        reverse = False

    # Define model matrix (identity for simplicity)
    model_matrix = np.identity(4)
    up = np.array([0, 1, 0])
    # Compute view and projection matrices
    view_matrix = look_at(camera_position, target, up)
    projection_matrix = perspective_projection(fov_number, aspect_ratio, near, far)
    # Define a 3D point in homogeneous coordinates
    valid_points = np.hstack((points_data[:, 1].reshape(-1, 1), points_data[:, 2].reshape(-1, 1), points_data[:, 0].reshape(-1, 1),
                        np.ones((points_data.shape[0], 1)), points_data[:, 3:], label.reshape(-1, 1)))  # (x, y, z, w)

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
        cv2.imwrite(save_image+'.jpg', image_rgb)

def gen_the_pp_image(scan, label, scan_path):
    for f in np.arange(3):
        number = f*30 - 30
        fov = np.pi / 1.2
        for h in np.arange(1):
            for w in np.arange(1):
                for l in np.arange(1):
                    save_label_path = "/home/xi/repo/research_2/PP/label/left_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                      scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image/left_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                             scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="left", save_image=save_image_path,
                                          save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number)

                    save_label_path = "/home/xi/repo/research_2/PP/label/right_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image/right_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="right", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number)

                    save_label_path = "/home/xi/repo/research_2/PP/label/forward_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image/forward_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="forward", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number)

                    save_label_path = "/home/xi/repo/research_2/PP/label/back_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3] + '.label'
                    save_image_path = "/home/xi/repo/research_2/PP/image/back_" + str(-number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                                 scan_path.split(os.sep)[-3]
                    do_perspective_projection(points_3d=scan, label=label, target_type="back", save_image=save_image_path,
                                              save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov, ang=number)

                    # save_label_path = "/home/xi/repo/research_2/PP/label/5_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                    #                   scan_path.split(os.sep)[-3] + '.label'
                    # save_image_path = "/home/xi/repo/research_2/PP/image/5_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                    #                   scan_path.split(os.sep)[-3]
                    # do_perspective_projection(points_3d=scan, label=label, target_type="up", save_image=save_image_path,
                    #                           save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov)
                    #
                    # save_label_path = "/home/xi/repo/research_2/PP/label/6_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                    #                   scan_path.split(os.sep)[-3] + '.label'
                    # save_image_path = "/home/xi/repo/research_2/PP/image/6_" + str(number) + '_' + str(h) + str(w) + str(l) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                    #                   scan_path.split(os.sep)[-3]
                    # do_perspective_projection(points_3d=scan, label=label, target_type="down", save_image=save_image_path,
                    #                           save_label=save_label_path, heights=h+1, widths=w+1, longitudes=l+1, fov_number=fov)


def get_the_pp_image(scan, label, scan_path):
    for h in np.arange(5):
        fov_value = np.pi / (1.2 + h * 0.5)
        view = 0
        while(view < 6):
            save_label_path_side = "/home/xi/repo/research_3/PP/label/side_" + str(h+1.2) + '_' + str(view) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                            scan_path.split(os.sep)[-3] + '.label'
            save_image_path_side = "/home/xi/repo/research_3/PP/image/side_" + str(h+1.2) + '_' + str(view) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                            scan_path.split(os.sep)[-3]
            do_side_pp_projection(points_3d=scan, label=label, save_image=save_image_path_side,
                            save_label=save_label_path_side, view=view, fov_value=fov_value)
            save_label_path_mid = "/home/xi/repo/research_3/PP/label/mid_" + str(h+1.2) + '_' + str(view) + '_' + \
                                  scan_path.split(os.sep)[-4] + '_' + \
                                  scan_path.split(os.sep)[-3] + '.label'
            save_image_path_mid = "/home/xi/repo/research_3/PP/image/mid_" + str(h+1.2) + '_' + str(view) + '_' + \
                                  scan_path.split(os.sep)[-4] + '_' + \
                                  scan_path.split(os.sep)[-3]
            do_mid_pp_projection(points_3d=scan, label=label, save_image=save_image_path_mid,
                                 save_label=save_label_path_mid, view=view, fov_value=fov_value)

            view += 1

def do_side_pp_projection(points_3d, label, save_image, save_label, view, fov_value):
    # Room Dimension
    X = np.max(points_3d[:, 1]) - np.min(points_3d[:, 1])
    Y = np.max(points_3d[:, 2]) - np.min(points_3d[:, 2])
    Z = np.max(points_3d[:, 0]) - np.min(points_3d[:, 0])

    target = np.array([np.mean(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.mean(points_3d[:, 0])])
    up = np.array([0, 1, 0])

    if (view == 0):
        camera_position = np.array([np.min(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.mean(points_3d[:, 0])])
    elif (view == 1):
        camera_position = np.array(
            [np.max(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.mean(points_3d[:, 0])])
    elif (view == 2):
        camera_position = np.array(
            [np.mean(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.min(points_3d[:, 0])])
    elif (view == 3):
        camera_position = np.array(
            [np.mean(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.max(points_3d[:, 0])])
    elif (view == 4):
        camera_position = np.array(
            [np.mean(points_3d[:, 1]), np.max(points_3d[:, 2]), np.mean(points_3d[:, 0])])
        up = np.array([0, 0, 1])
    elif (view == 5):
        camera_position = np.array(
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
    points = np.hstack((points_3d[:, 1].reshape(-1, 1), points_3d[:, 2].reshape(-1, 1), points_3d[:, 0].reshape(-1, 1),
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
    np.savetxt(save_label, proj_l)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_image + '.jpg', image_rgb)

def do_mid_pp_projection(points_3d, label, save_image, save_label, view, fov_value):
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
    points = np.hstack((points_3d[:, 1].reshape(-1, 1), points_3d[:, 2].reshape(-1, 1), points_3d[:, 0].reshape(-1, 1),
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
    np.savetxt(save_label, proj_l)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_image + '.jpg', image_rgb)