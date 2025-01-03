# Pre-define the view params (location, view angle... ,etc.)
width, height = 512, 512
# fov = np.pi / 1.4  # 60 degrees
near, far = 0.1, 1000
aspect_ratio = width / height

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(num_classes=14).to(device)
if torch.cuda.device_count() >= 1:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
    model = model.to(device)
    model = model.cuda()
model.load_state_dict(torch.load('/home/xi/repo/VGG/log/model_UNET_PP3_LR1.2-04.pth'))
model.eval()


IoU_res = []
# ------------------------------------------------------------------------------------
base_dir = '/home/xi/repo/Stanford3dDataset_v1.2_Aligned_Version_Test/'
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
    l = gen_the_pp_image(scan=scan, label=label, scan_path=scan_path)

    bj = pj(scan)
    extended_points_with_labels = bj.predict_labels_weighted(l)
    # extended_points_with_labels = bj.predic_labels_appearance(l)
    # extended_points_with_labels = bj.predict_labels(image_mask, proj_W=512, proj_H=512, proj_fov_up=110, proj_fov_down = -110)
    _, _, _, IoU = get_3d_eval_res(extended_points_with_labels[:, -1], label)

    IoU_res.append(IoU)


print(np.mean(IoU_res))