
import os
import numpy as np
import open3d as o3d

def vis(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(color / 255)
    pcd.estimate_normals()
    o3d.visualization.draw_geometries([pcd], window_name="mesh pcd")

label_dic = {
        0: "Unknown",
        1: "Cart",
        2: "Ladder",
        3: "Locker",
        4: "Rack",
        5: "Round",
        6: "Square",
        7: "Stairs",
        8: "Table",
        9: "Workbench",
        10: "Cleaning",
        11: "Wood Storage",
        12: "Vehical",
        13: "Box",
        14: "Wood"}

# data = np.loadtxt("/home/rosie/repo/Factory/Section 5/'Cart_15'_000004.pts")

folder_path = "/home/rosie/repo/Factory/Section 6"
pts_files = [os.path.join(root, file)
             for root, _, files in os.walk(folder_path)
             for file in files if file.endswith(".pts")]

data_room = np.zeros((0, 6))
data_room_label = np.zeros((0))
for i in pts_files:
    data = np.loadtxt(i)
    data_room = np.concatenate((data_room, data[:, 0:6]), axis=0)
    for key,name in label_dic.items():
        if name in i:
            label_value = np.full(data.shape[0], key, dtype=np.int32)
            data_room_label = np.concatenate((data_room_label, label_value), axis=0)
            break
    else:
        label_value = np.full(data.shape[0], 15, dtype=np.int32)
        data_room_label = np.concatenate((data_room_label, label_value), axis=0)
            
np.savetxt("/home/rosie/repo/Factory/section6.txt", data_room)
np.savetxt("/home/rosie/repo/Factory/section6.label", data_room_label)


