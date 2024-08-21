import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sympy.strategies.core import switch


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


def transform_point(point, model_matrix, view_matrix, projection_matrix):
    world_coords = np.dot(model_matrix, point[0:4, :])
    camera_coords = np.dot(view_matrix, world_coords)
    clip_coords = np.dot(projection_matrix, camera_coords)
    # ndc_coords = clip_coords[:3, :] / clip_coords[3, :]  # Perspective division
    ndc_coords = np.vstack(
        ((clip_coords[:3, :] / clip_coords[3, :]), point[4, :], point[5, :], point[6, :], point[7, :]))
    return np.transpose(ndc_coords)

# Pre-define the view params (location, view angle... ,etc.)
width, height = 224, 224
fov = np.pi / 3.0  # 60 degrees
near, far = 0.1, 1000
aspect_ratio = width / height

def do_perspective_projection(points_3d, label, target_type, save_image, save_label):
    # Room Dimension
    X = np.max(points_3d[:,1]) - np.min(points_3d[:,1])
    Y = np.max(points_3d[:,2]) - np.min(points_3d[:,2])
    Z = np.max(points_3d[:,0]) - np.min(points_3d[:,0])

    target = np.array([np.mean(points_3d[:,1]), np.mean(points_3d[:,2]), np.mean(points_3d[:,0])])
    if (target_type == "left"):
        camera_position = np.array([np.min(points_3d[:,1]), np.mean(points_3d[:,2]), np.mean(points_3d[:,0])])
    elif (target_type == "right"):
        camera_position = np.array(
            [np.max(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.mean(points_3d[:, 0])])
    elif (target_type == "forward"):
        camera_position = np.array(
            [np.mean(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.min(points_3d[:, 0])])
    else:
        camera_position = np.array(
            [np.mean(points_3d[:, 1]), np.mean(points_3d[:, 2]), np.max(points_3d[:, 0])])

    up = np.array([0, 1, 0])

    # Define model matrix (identity for simplicity)
    model_matrix = np.identity(4)

    # Compute view and projection matrices
    view_matrix = look_at(camera_position, target, up)
    projection_matrix = perspective_projection(fov, aspect_ratio, near, far)
    # Define a 3D point in homogeneous coordinates
    points = np.hstack((points_3d[:, 1].reshape(-1, 1), points_3d[:, 2].reshape(-1, 1), points_3d[:, 0].reshape(-1, 1),
                        np.ones((points_3d.shape[0], 1)), points_3d[:, 3:], label.reshape(-1, 1)))  # (x, y, z, w)

    # filter out the 3d points behind the camera and looking direction
    # forward = np.float_(target - camera_position)
    # forward /= np.linalg.norm(forward)
    # dot_products = np.dot(points[:,0:3], forward)
    # mask = dot_products > 0
    # valid_points = points[mask]

    valid_points = points
    # Transform the point through the whole pipeline
    new_coords = transform_point(np.transpose(valid_points), model_matrix, view_matrix, projection_matrix)

    # Project to screen coordinates
    screen_coords = project_to_screen(np.array(new_coords), width, height)

    x_coords = screen_coords[:, 0].astype(np.int32)

    y_coords = screen_coords[:, 1].astype(np.int32)

    colors = screen_coords[:, 3:6].astype(np.uint8)  # Assuming RGB values in uint8 format

    labels = screen_coords[:, 6:].astype(np.uint8)  # Assuming RGB values in uint8 format

    # Calculate the actual dimensions of the image
    img_width = min(np.max(x_coords) + 1, width)
    img_height = min(np.max(y_coords) + 1, height)

    # Create a blank image with specified dimensions
    image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    proj_l = np.zeros((img_height, img_width), dtype=np.uint8)
    # Assign colors to corresponding coordinates within the limits
    valid_indices = (x_coords < img_width) & (y_coords < img_height) & (0 <= x_coords) & (0 <= y_coords)
    image[y_coords[valid_indices], x_coords[valid_indices]] = colors[valid_indices]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_image+'.png', image_rgb)

    proj_l[y_coords[valid_indices], x_coords[valid_indices]] = labels[valid_indices].reshape(-1)
    np.savetxt(save_label, proj_l)