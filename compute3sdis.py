import os
import numpy as np


#dictionary for calculating the points number in each category
label_dic = {
        0: "clutter",
        1: "beam",
        2: "column",
        3: "wall",
        4: "ceiling",
        5: "floor",
        6: "bookcase",
        7: "door",
        8: "chair",
        9: "board",
        10: "sofa",
        11: "window",
        12: "table",
        13: "stairs"}

catPointCount = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 0,
        6: 0,
        7: 0,
        8: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0}
# label_dic = {
#         0: "clutter",
#         1: "wall",
#         2: "ceiling",
#         3: "floor",
#         4: "door"}
#
# catPointCount = {
#         0: 0,
#         1: 0,
#         2: 0,
#         3: 0,
#         4: 0}
# label_dic = {
#         0: "clutter",
#         1: "column",
#         2: "wall",
#         3: "ceiling",
#         4: "floor",
#         5: "bookcase",
#         6: "door",
#         7: "chair",
#         8: "board",
#         9: "stairs"}
#
# catPointCount = {
#         0: 0,
#         1: 0,
#         2: 0,
#         3: 0,
#         4: 0,
#         5: 0,
#         6: 0,
#         7: 0,
#         8: 0,
#         9: 0}

pointcount = []

means_x = []
means_y = []
means_z = []
means_r = []
means_g = []
means_b = []
means_intensity = []

std_x = []
std_y = []
std_z = []
std_r = []
std_g = []
std_b = []
std_intensity = []


#idx = [3]
idx = [6]
for n in idx:
    path = "/home/xi/repo/3sdis/Stanford3dDataset_v1.2_Aligned_Version/Area_"+str(n)
    fdirs = os.listdir(path)
    for fdir in fdirs:
        path_l2 = os.path.join(path, fdir)
        if os.path.isdir(path_l2):
            # if "label" in fdir or "room_data" in fdir
            print("first path_l2: " + path_l2)
            # Generate the overall room data and label
            data_room = np.zeros((0, 6))
            data_room_label = np.zeros((0))
            for p,d,f in os.walk(path_l2 + "/Annotations"):
                txtFileNum = 0
                for fp in f:
                    if fp.endswith(".txt"):
                        pointfile = os.path.join(p, fp)
                        print(pointfile)
                        data = np.loadtxt(pointfile, dtype=np.float32).reshape(-1,6)
                        txtFileNum += 1
                        data_room = np.concatenate((data_room, data), axis=0)
                        depth = np.linalg.norm(data[:,0:3], 2, axis=1)
                        pointcount.append(data.shape[0])
                        means_r.append(np.mean(data[:,3]))
                        means_g.append(np.mean(data[:,4]))
                        means_b.append(np.mean(data[:,5]))
                        means_x.append(np.mean(data[:,0]))
                        means_y.append(np.mean(data[:,1]))
                        means_z.append(np.mean(data[:,2]))
                        means_intensity.append(np.mean(depth))
        
                        std_x.append(np.std(data[:,0]))
                        std_y.append(np.std(data[:,1]))
                        std_z.append(np.std(data[:,2]))
                        std_r.append(np.std(data[:,3]))
                        std_g.append(np.std(data[:,4]))
                        std_b.append(np.std(data[:,5]))
                        std_intensity.append(np.std(depth))


                        for key,name in label_dic.items():
                            if name in fp:
                                catPointCount[key] += data.shape[0]
                                label_value = np.full(data.shape[0], key, dtype=np.int32)
                                data_room_label = np.concatenate((data_room_label, label_value), axis=0)
                                #Generate the label if doesn't exist
                                label_path = path_l2 + "/label"
                                print("make folder" + label_path + "we have path_l2:" + path_l2)
                                os.makedirs(label_path, exist_ok=True)
                                labels_num = len(os.listdir(label_path))
                                files_list = os.listdir(path_l2 + "/Annotations")
                                files_num = 0
                                for i in files_list:
                                    if i.endswith(".txt"):
                                        files_num += 1
                                if (files_num != labels_num):
                                    label_name = fp.split(".txt", 1)[0] + ".label"
                                    # print(label_path+"/"+label_name)
                                    np.savetxt(label_path + "/" + label_name, label_value)
                # print(txtFileNum)
            # if not os.path.isdir(path_l2 + "/room_data"):
            # print("make folder" + path_l2 + "/room_data now")
            os.makedirs(path_l2 + "/room_data", exist_ok=True)
            np.savetxt(path_l2 + "/room_data/" + fdir + ".label", data_room_label)
            np.savetxt(path_l2 + "/room_data/" + fdir + ".txt", data_room)

allpoints = sum(pointcount)
mX_array = np.array(means_x) * np.array(pointcount)
mY_array = np.array(means_y) * np.array(pointcount)
mZ_array = np.array(means_z) * np.array(pointcount)
mI_array = np.array(means_intensity) * np.array(pointcount)
mR_array = np.array(means_r) * np.array(pointcount)
mG_array = np.array(means_g) * np.array(pointcount)
mB_array = np.array(means_b) * np.array(pointcount)



mX = sum(mX_array) / allpoints
mY = sum(mY_array) / allpoints
mZ = sum(mZ_array) / allpoints
mI = sum(mI_array) / allpoints
mR = sum(mR_array) / allpoints
mG = sum(mG_array) / allpoints
mB = sum(mB_array) / allpoints


sX_array = np.array(std_x) * np.array(pointcount)
sY_array = np.array(std_y) * np.array(pointcount)
sZ_array = np.array(std_z) * np.array(pointcount)
sR_array = np.array(std_r) * np.array(pointcount)
sG_array = np.array(std_g) * np.array(pointcount)
sB_array = np.array(std_b) * np.array(pointcount)
sI_array = np.array(std_intensity) * np.array(pointcount)

sR = sum(sR_array) / allpoints
sG = sum(sG_array) / allpoints
sB = sum(sB_array) / allpoints
sX = sum(sX_array) / allpoints
sY = sum(sY_array) / allpoints
sZ = sum(sZ_array) / allpoints
sI = sum(sI_array) / allpoints

# print("mI = %f" % (mI))
print("mX = %f" % (mX))
print("mY = %f" % (mY))
print("mZ = %f" % (mZ))
print("mR = %f" % (mR))
print("mG = %f" % (mG))
print("mB = %f" % (mB))


print("sI = %f" % (sI))
print("sX = %f" % (sX))
print("sY = %f" % (sY))
print("sZ = %f" % (sZ))
print("sR = %f" % (sR))
print("sG = %f" % (sG))
print("sB = %f" % (sB))
#
for cat,num in catPointCount.items():
    print("the category is %d and the ratio of points number is equal to %f" % (cat, num))
