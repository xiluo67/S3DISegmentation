import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from numpy.core.memmap import dtypedescr
from sympy.physics.units.definitions.dimension_definitions import angle

"""
This file is used for Spherical Projection and Birds Eye View Projection: projecting 3D points to 2D plane
----------------------------------------------------------------------------------------------------------
Date: 2024.11

"""

# Load the dataset file
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


# Define the base directory
# base_dir = '/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/'
# base_dir = '/home/xi/repo/Stanford3dDataset_v1.2_Aligned_Version_Test/'
base_dir = '/home/xi/repo/new'


# Find all .txt files
scan_files, scan_labels = find_txt_files(base_dir)

# Print all .txt file paths
for txt_file in scan_files:
    print(txt_file)

print(len(scan_labels))
print(len(scan_files))
import numpy as np


def calculate_yaw(x, y, theta=0):
    # 旋转坐标系，调整起始方向
    x_prime = x * np.cos(theta) - y * np.sin(theta)
    y_prime = x * np.sin(theta) + y * np.cos(theta)

    # # 计算yaw角
    # yaw = -np.arctan2(y_prime, x_prime)
    #
    # # 计算投影位置
    # proj_x = 0.5 * (yaw / np.pi + 1.0)  # 范围 [0.0, 1.0]

    return y_prime, x_prime

# theta = np.radians(45)  # 45度角
def do_range_projection(points, proj_fov_up, proj_fov_down, save_image, save_label, proj_W, proj_H, R, G, B, label, angle):
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


    # get angles of all points
    if gen == 1:
        # get depth of all points
        depth_points = np.zeros(points.shape)
        depth_points[:, 0] = scan_x - np.mean(scan_x)
        depth_points[:, 1] = scan_y - np.mean(scan_y)
        depth_points[:, 2] = scan_z - np.mean(scan_z)
        depth = np.linalg.norm(depth_points, 2, axis=1)
        yaw = -np.arctan2(depth_points[:, 1], depth_points[:, 0]) + np.pi / 4
        pitch = np.arcsin(depth_points[:, 2] / depth)
    elif gen == 0:
        depth_points = np.zeros(points.shape)
        depth_points[:, 0] = scan_x - np.mean(scan_x)
        depth_points[:, 1] = scan_y - np.mean(scan_y)
        depth_points[:, 2] = scan_z - np.mean(scan_z)
        depth = np.linalg.norm(depth_points, 2, axis=1)
        yaw = -np.arctan2(depth_points[:, 1], depth_points[:, 0]) + np.pi / 2
        pitch = np.arcsin(depth_points[:, 2] / depth)

    elif gen == 2:
        depth_points = np.zeros(points.shape)
        depth_points[:, 0] = scan_x - np.mean(scan_x)
        depth_points[:, 1] = scan_y - np.mean(scan_y)
        depth_points[:, 2] = scan_z - np.mean(scan_z)
        depth = np.linalg.norm(depth_points, 2, axis=1)
        yaw = -np.arctan2(depth_points[:, 1], depth_points[:, 0]) - np.pi / 2
        pitch = np.arcsin(depth_points[:, 2] / depth)

    elif gen == 3:
        depth_points = np.zeros(points.shape)
        depth_points[:, 0] = scan_x - np.mean(scan_x)
        depth_points[:, 1] = scan_y - np.mean(scan_y)
        depth_points[:, 2] = scan_z - np.mean(scan_z)
        depth = np.linalg.norm(depth_points, 2, axis=1)
        yaw = -np.arctan2(depth_points[:, 1], depth_points[:, 0]) - np.pi / 4
        pitch = np.arcsin(depth_points[:, 2] / depth)

    elif gen == 4:
        depth_points = np.zeros(points.shape)
        depth_points[:, 0] = scan_x - np.mean(scan_x)
        depth_points[:, 1] = scan_y - np.mean(scan_y)
        depth_points[:, 2] = scan_z - np.mean(scan_z)
        depth = np.linalg.norm(depth_points, 2, axis=1)
        yaw = -np.arctan2(depth_points[:, 1], depth_points[:, 0])
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
    proj_xyz[proj_y, proj_x] = points
    proj_idx[proj_y, proj_x] = indices
    proj_r[proj_y, proj_x] = R[order]
    proj_g[proj_y, proj_x] = G[order]
    proj_b[proj_y, proj_x] = B[order]
    proj_l[proj_y, proj_x] = label[order]
    proj_mask = (proj_idx > 0).astype(np.int32)

    num_colors = np.max(proj_l) + 1

    # Define a colormap with the specified number of colors
    base_colormap = plt.get_cmap('viridis')
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
    plt.imshow(proj)
    plt.axis('off')
    plt.savefig(save_image+'.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)

def BEV_Height(points, height_range, res, proj_W, proj_H):

    points = points
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    fwd_range = (np.min(x_points), np.max(x_points))
    side_range = (np.min(y_points), np.max(y_points))
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    v_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filter = np.logical_and(f_filt, v_filt)
    indices = np.argwhere(filter).flatten()

    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    # R = R[indices]
    # G = G[indices]
    # B = B[indices]
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
    im = -10 * np.ones([proj_H, proj_W], dtype=np.uint8)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    ia = indices.shape[0]
    for i in np.arange(ia):
        y = int((y_img[i] / x_max) * proj_W)
        x = int((x_img[i] / y_max) * proj_H)
        if im[x, y] <= height_values[i]:
            im[x, y] = height_values[i]

    return im

def point_cloud_2_birdseye(save_image, save_label, height_range, points, R,G,B,label, proj_W=512, proj_H=512, res=0.005):
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
    depth = np.linalg.norm(depth_points, 2, axis=1)
    unproj_range = np.copy(depth)

    # BEV Projection
    fwd_range = (np.min(x_points), np.max(x_points))
    side_range = (np.min(y_points), np.max(y_points))

    height_range_1 = (np.min(z_points), np.max(z_points) / 4)
    height_range_2 = (np.max(z_points)/4, np.max(z_points))

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > side_range[0]), (y_points < side_range[1]))
    v_filt = np.logical_and((z_points > height_range[0]), (z_points < height_range[1]))
    filter = np.logical_and(f_filt, v_filt)
    indices = np.argwhere(filter).flatten()

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
    #y_img += int(np.ceil(fwd_range[1] / res))

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
    im = -10 * np.ones([proj_H, proj_W], dtype=np.uint8)

    proj_range = -10 * np.ones([proj_H, proj_W], dtype=np.uint8)
    proj_r = np.zeros([proj_H, proj_W], dtype=np.uint8)
    proj_g = -10 * np.zeros([proj_H, proj_W], dtype=np.uint8)
    proj_b = -10 * np.zeros([proj_H, proj_W], dtype=np.uint8)
    proj_idx = -10 * np.zeros([proj_H, proj_W], dtype=np.uint8)
    proj_l = -10 * np.zeros([proj_H, proj_W], dtype=np.uint8)
    proj_xyz = -10 * np.zeros([proj_H, proj_W, 3], dtype=np.uint8)
    # FILL PIXEL VALUES IN IMAGE ARRAY
    ia = indices.shape[0]
    for i in np.arange(ia):
        y = int((y_img[i] / x_max) * proj_W)
        x = int((x_img[i] / y_max) * proj_H)
        proj_range[x, y] += 1
        proj_r[x, y] = R[i]
        proj_g[x, y] = G[i]
        proj_b[x, y] = B[i]
        proj_idx[x, y] = indices[i]
        proj_l[x, y] = label[i]
        proj_xyz[x, y, 0] = np.maximum(im[x, y], height_values[i])

    proj_xyz[:, :, 1] = BEV_Height(points, height_range_1, res, proj_W, proj_H)
    proj_xyz[:, :, 2] = BEV_Height(points, height_range_2, res, proj_W, proj_H)
    temp_min = np.min(proj_range)
    temp_max = np.max(proj_range)
    for co in np.arange(proj_range.shape[1]):
        proj_range[:, co] = scale_to_255(proj_range[:, co], temp_min, temp_max)

    #show image
    proj = np.zeros([proj_H, proj_W, 3], dtype=np.uint8)
    proj[:, :, 0] = proj_r
    proj[:, :, 1] = proj_g
    proj[:, :, 2] = proj_b

    num_colors = np.max(proj_l) + 1
    # Define a colormap with the specified number of colors
    base_colormap = plt.get_cmap('viridis')
    colormap = base_colormap(np.linspace(0, 1, num_colors))
    norm = mcolors.Normalize(vmin=np.min(proj_l), vmax=np.max(proj_l))
    colorized_image = mcolors.ListedColormap(colormap)(norm(proj_l))

    plt.imshow(colorized_image)
    plt.axis('off')
    # plt.savefig('lounge_label.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.imshow(proj)
    plt.axis('off')
    plt.savefig(save_image+'.jpg', bbox_inches='tight', pad_inches=0)
    plt.close()
    np.savetxt(save_label, proj_l)
    # assing to images
    proj_x = points[:, 1] / np.max(points[:, 1]) * proj_W
    proj_y = points[:, 0] / np.max(points[:, 0]) * proj_H
    proj_mask = (proj_idx > 0).astype(np.int32)

from PP import *

name = "SP"
# if all goes well, open point cloud
for i in np.arange(len(scan_files)):
    scan_path = scan_files[i]
    label_path = scan_labels[i]

    scan = np.loadtxt(scan_path, dtype=np.float32)
    scan = scan.reshape((-1, 6))

    # put in attribute
    points = scan[:, 0:3]  # get xyz
    R = scan[:, 3]  # get R
    G = scan[:, 4]  # get G
    B = scan[:, 5]  # get B
    label = np.loadtxt(label_path, dtype=np.float32)
    if (name == "SP"):
        gen = 0
        save_label = "/home/xi/repo/research_3/SP/label/0_" + scan_path.split(os.sep)[-4] + '_' + \
                     scan_path.split(os.sep)[-3] + '.label'
        save_image = "/home/xi/repo/research_3/SP/image/0_" + scan_path.split(os.sep)[-4] + '_' + \
                     scan_path.split(os.sep)[-3]
        do_range_projection(save_image=save_image, save_label=save_label, points=points, proj_fov_up=110,
                            proj_fov_down=-110, proj_W=512,
                            proj_H=512, R=R, G=G, B=B)
        gen = 1
        save_label = "/home/xi/repo/research_3/SP/label/1_" + scan_path.split(os.sep)[-4] + '_' + scan_path.split(os.sep)[-3] + '.label'
        save_image = "/home/xi/repo/research_3/SP/image/1_" + scan_path.split(os.sep)[-4] + '_' + scan_path.split(os.sep)[-3]
        do_range_projection(save_image=save_image, save_label=save_label, points=points, proj_fov_up=110, proj_fov_down=-110, proj_W=512,
                            proj_H=512, R=R, G=G, B=B)
        gen = 2
        save_label = "/home/xi/repo/research_3/SP/label/2_" + scan_path.split(os.sep)[-4] + '_' + \
                     scan_path.split(os.sep)[-3] + '.label'
        save_image = "/home/xi/repo/research_3/SP/image/2_" + scan_path.split(os.sep)[-4] + '_' + \
                     scan_path.split(os.sep)[-3]
        do_range_projection(save_image=save_image, save_label=save_label, points=points, proj_fov_up=110,
                            proj_fov_down=-110, proj_W=512,
                            proj_H=512, R=R, G=G, B=B)
        gen = 3
        save_label = "/home/xi/repo/research_3/SP/label/3_" + scan_path.split(os.sep)[-4] + '_' + \
                     scan_path.split(os.sep)[-3] + '.label'
        save_image = "/home/xi/repo/research_3/SP/image/3_" + scan_path.split(os.sep)[-4] + '_' + \
                     scan_path.split(os.sep)[-3]
        do_range_projection(save_image=save_image, save_label=save_label, points=points, proj_fov_up=110,
                            proj_fov_down=-110, proj_W=512,
                            proj_H=512, R=R, G=G, B=B)
        gen = 4
        save_label = "/home/xi/repo/research_3/SP/label/4_" + scan_path.split(os.sep)[-4] + '_' + \
                     scan_path.split(os.sep)[-3] + '.label'
        save_image = "/home/xi/repo/research_3/SP/image/4_" + scan_path.split(os.sep)[-4] + '_' + \
                     scan_path.split(os.sep)[-3]
        do_range_projection(save_image=save_image, save_label=save_label, points=points, proj_fov_up=110,
                            proj_fov_down=-110, proj_W=512,
                            proj_H=512, R=R, G=G, B=B)

    elif (name == "BEV"):
        hli = (np.max(points[:, 2]) - np.min(points[:, 2])) / 7
        for i in range(6):
            start = hli * i
            end = hli * (i+1)
            height_range = (np.min(points[:, 2]) + start, np.min(points[:, 2]) + end)
            save_label = "/home/xi/repo/research_Area5/BEV/label/" + str(i) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                        scan_path.split(os.sep)[-3] + '.label'
            save_image = "/home/xi/repo/research_Area5/BEV/image/" + str(i) + '_' + scan_path.split(os.sep)[-4] + '_' + \
                        scan_path.split(os.sep)[-3]
            point_cloud_2_birdseye(save_image=save_image, save_label=save_label, height_range=height_range, points=points, R=R,G=G,B=B,label=label)

    elif (name == "PP"):
        gen_the_pp_image(scan=scan, label=label, scan_path=scan_path)
    else:
        pass

# scp = "/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_1/room_data/auditorium_1.txt"
# lap = "/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_2/auditorium_1/room_data/auditorium_1.label"
#
# sc = np.loadtxt(scp, dtype=np.float32)
# sc = sc.reshape((-1, 6))
# # put in attribute
# la = np.loadtxt(lap, dtype=np.float32)
#
# gen_the_pp_image(scan=sc, label=la, scan_path=scp)
