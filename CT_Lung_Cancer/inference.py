import os
import cv2
import torch
import numpy as np
import pandas as pd
import nibabel as nib
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import math
from copy import deepcopy
import argparse
from src.model.inception_resnet_v2.multi_task.multi_task_3d import InceptionResNetV2MultiTask3D

CLIP_MIN, CLIP_MAX = -1000, 200
in_channels = 1
num_classes = 2
size = 96
BASE_DTYPE = torch.float32
DATA_SIZE = 512
# center point needs to be [x, y, z]
def get_center_idx_from_point(sitk_obj, center_point):
    center_idx = sitk_obj.TransformPhysicalPointToIndex(center_point)
    
    col_idx, row_idx, z_idx = center_idx
    
    return z_idx, row_idx, col_idx

def get_slices_and_padding(center_idx, patch_size, img_shape):
    half_size = patch_size // 2
    z_num, x_num, y_num = img_shape
    
    center_idx = list(center_idx)
    center_idx[0] = min(center_idx[0], z_num - half_size) 

    z_slice = slice(max(0, center_idx[0]-half_size),
                    min(z_num, center_idx[0]+half_size)) 
    x_slice = slice(max(0, center_idx[1]-half_size),
                    min(x_num, center_idx[1]+half_size)) 
    y_slice = slice(max(0, center_idx[2]-half_size),
                    min(y_num, center_idx[2]+half_size)) 

    z_pad_num = patch_size - (z_slice.stop - z_slice.start)
    x_pad_num = patch_size - (x_slice.stop - x_slice.start)
    y_pad_num = patch_size - (y_slice.stop - y_slice.start)

    z_left_pad = z_pad_num // 2
    x_left_pad = x_pad_num // 2
    y_left_pad = y_pad_num // 2
    
    z_right_pad = z_left_pad + z_pad_num % 2
    x_right_pad = x_left_pad + x_pad_num % 2
    y_right_pad = y_left_pad + y_pad_num % 2

    return z_slice, x_slice, y_slice, (z_left_pad, z_right_pad), (x_left_pad, x_right_pad), (y_left_pad, y_right_pad)


def get_patch_from_center_idx(img_array, center_idx, patch_size=96):
    z_slice, x_slice, y_slice, z_padding, x_padding, y_padding = get_slices_and_padding(center_idx, patch_size, img_array.shape)
    patch = img_array[z_slice, x_slice, y_slice]
    padded_patch = np.pad(patch, (z_padding, x_padding, y_padding), mode='constant', constant_values=0)
    return padded_patch

    
def fill_value_in_mask(mask_array, patch_mask_array, center_idx, patch_size=96):
    z_slice, x_slice, y_slice, z_padding, x_padding, y_padding = get_slices_and_padding(center_idx, patch_size, mask_array.shape)
    
    z_left_pad, z_right_pad = z_padding
    x_left_pad, x_right_pad = x_padding
    y_left_pad, y_right_pad = y_padding
    
    z_right_pad = None if z_right_pad == 0 else -z_right_pad 
    x_right_pad = None if x_right_pad == 0 else -x_right_pad
    y_right_pad = None if y_right_pad == 0 else -y_right_pad
    
    previous_slice = mask_array[z_slice, x_slice, y_slice]
    target_slice = patch_mask_array[slice(z_left_pad, z_right_pad),
                                    slice(x_left_pad, x_right_pad),
                                    slice(y_left_pad, y_right_pad)]
    replace_slice = np.maximum(previous_slice, target_slice)
    mask_array[z_slice, x_slice, y_slice] = replace_slice
    return mask_array

def preprocess(img_array):
    global CLIP_MIN, CLIP_MAX
    img_array = np.clip(img_array, CLIP_MIN, CLIP_MAX)
    img_array = (img_array - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)
    return np.round(img_array * 255).astype("uint8")

def save_center_pred_plots(center_point, sitk_array, pred_array, shape_idx, save_path, resize=False):
    # 사용자 정의 colormap 생성
    colors = [(1, 1, 1, 0), (1, 0, 0)]  # 시작: 투명한 흰색, 끝: 빨간색
    cmap_name = 'custom_div_cmap'
    cm = LinearSegmentedColormap.from_list(
            cmap_name, colors, N=100)
    
    center_point = list(center_point)
    target_idx = center_point.pop(shape_idx) 
    x, y = center_point[1], center_point[0] 
    img_slice = np.take(sitk_array, indices=target_idx, axis=shape_idx).clip(CLIP_MIN, CLIP_MAX)
    pred_slice = np.take(pred_array, indices=target_idx, axis=shape_idx)
    if resize:
        y = int(round(y * 512 / img_slice.shape[0]))
        img_slice = cv2.resize(img_slice, (512, 512), interpolation=cv2.INTER_LINEAR_EXACT)
        pred_slice = cv2.resize(pred_slice, (512, 512), interpolation=cv2.INTER_NEAREST)

    fig, ax = plt.subplots(1, 3, figsize=(24, 8))
    fig.dpi = 500  # 여기서 DPI를 설정
    ax[0].imshow(img_slice, cmap="gray")
    ax[1].imshow(img_slice, cmap="gray")
    ax[1].imshow(pred_slice, cmap=cm, alpha=0.25)
    ax[1].scatter(x, y, color='blue', marker='*')
    ax[2].imshow(pred_slice,cmap="gray")
    
    plt.tight_layout()  # 그림 주변의 여백 제거
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.5)  # 저장할 때도 여백 제거
    plt.close()
    
def save_numpy_to_nifti(data: np.ndarray, filename: str, spacing: tuple = (1,1,1), origin: tuple = (0,0,0)):
    # Validate the input data
    if data.ndim != 3:
        raise ValueError("Input data should be a 3D numpy array.")
    
    # Convert the Numpy array to a SimpleITK image
    img = sitk.GetImageFromArray(data)
    
    # Set metadata
    img.SetSpacing(spacing)
    img.SetOrigin(origin)

    # Save the image to a NIfTI file
    sitk.WriteImage(img, filename)

def load_model(weight_path, model):
    state_dict = torch.load(weight_path)
    state_dict = {key.replace("module.", ""): value
                 for key, value in state_dict.items()}
    model.load_state_dict(state_dict)

def get_1x1x1_patch_from_center_idx(img_array, center_idx, patch_size=96):
    Z, H, W = img_array.shape
    patch_list = []
    center_idx_list = []
    for z_idx in range(0, 1):
        for row_idx in range(0, 1):
            for col_idx in range(0, 1):
                center_idx_part = (np.clip(center_idx[0] + z_idx * patch_size, 0, Z), 
                                   np.clip(center_idx[1] + row_idx * patch_size, 0, H), 
                                   np.clip(center_idx[2] + col_idx * patch_size, 0, W))
                z_slice, x_slice, y_slice, z_padding, x_padding, y_padding = get_slices_and_padding(center_idx_part, 
                                                                                                    patch_size, img_array.shape)
                patch = img_array[z_slice, x_slice, y_slice]
                padded_patch = np.pad(patch, (z_padding, x_padding, y_padding), mode='constant', constant_values=0)
                patch_list.append(padded_patch)
                center_idx_list.append(center_idx_part)
    return np.stack(patch_list, axis=0), center_idx_list

def get_3x3x3_patch_from_center_idx(img_array, center_idx, patch_size=96):
    Z, H, W = img_array.shape
    patch_list = []
    center_idx_list = []
    for z_idx in range(-1, 2):
        for row_idx in range(-1, 2):
            for col_idx in range(-1, 2):
                center_idx_part = (np.clip(center_idx[0] + z_idx * patch_size, 0, Z), 
                                   np.clip(center_idx[1] + row_idx * patch_size, 0, H), 
                                   np.clip(center_idx[2] + col_idx * patch_size, 0, W))
                z_slice, x_slice, y_slice, z_padding, x_padding, y_padding = get_slices_and_padding(center_idx_part, 
                                                                                                    patch_size, img_array.shape)
                patch = img_array[z_slice, x_slice, y_slice]
                padded_patch = np.pad(patch, (z_padding, x_padding, y_padding), mode='constant', constant_values=0)
                patch_list.append(padded_patch)
                center_idx_list.append(center_idx_part)
    return np.stack(patch_list, axis=0), center_idx_list

def get_5x5x5_patch_from_center_idx(img_array, center_idx, patch_size=96):
    Z, H, W = img_array.shape
    stride = patch_size // 2
    patch_list = []
    center_idx_list = []
    for z_idx in range(-2, 3):
        for row_idx in range(-2, 3):
            for col_idx in range(-2, 3):
                center_idx_part = (np.clip(center_idx[0] + z_idx * stride, 0, Z), 
                                   np.clip(center_idx[1] + row_idx * stride, 0, H), 
                                   np.clip(center_idx[2] + col_idx * stride, 0, W))
                z_slice, x_slice, y_slice, z_padding, x_padding, y_padding = get_slices_and_padding(center_idx_part, 
                                                                                                    patch_size, img_array.shape)
                patch = img_array[z_slice, x_slice, y_slice]
                padded_patch = np.pad(patch, (z_padding, x_padding, y_padding), mode='constant', constant_values=0)
                patch_list.append(padded_patch)
                center_idx_list.append(center_idx_part)
    return np.stack(patch_list, axis=0), center_idx_list

def process_patch_array(patch_tensor, target_model, batch_size=4, threshold=0.5):
    data_num = patch_tensor.shape[0]
    batch_num = math.ceil(data_num / batch_size) 
    pred_patch_array = []
    for batch_idx in range(batch_num): 
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, data_num)
        pred_patch_array_part = target_model(patch_tensor[start_idx:end_idx])
        if isinstance(pred_patch_array_part, list):
            pred_patch_array_part = pred_patch_array_part[0]
        pred_patch_array.append(pred_patch_array_part)
    pred_patch_array = torch.cat(pred_patch_array, axis=0)
    if pred_patch_array.size(1) == 1:
        pred_patch_array = pred_patch_array[:, 0]        
    else:
        pred_patch_array = pred_patch_array[:, 1]
    pred_patch_array = (pred_patch_array > threshold).float()
    return pred_patch_array


parser = argparse.ArgumentParser(description='Lung Cancer Segmentation Module Following Manual Center Point')

parser.add_argument('--target_dir', type=str, default="./data", help='The target directory path')
parser.add_argument('--output_dir', type=str, default="./output", help="The directory that results are saved")
parser.add_argument('--ext', type=str, default=".nii", help="file extension you use. you can choose nii, nii.gz")
parser.add_argument('--gpu_num', type=int, default=-1, help="The gpu num you use. -1 means use cpu")
parser.add_argument('--dtype', type=str, default="float16", help="dtype you use. you can choose float16, float32, float64, default is float16")
parser.add_argument('--weight_path', type=str, default="./model_weights/model_multi_best.ckpt", help="The weight ckpt path you use")
parser.add_argument('--mask_threshold', type=float, default=0.5, help="mask thresold")

args = parser.parse_args()

# 받아온 인자 사용
target_dir = args.target_dir
output_dir = args.output_dir
target_ext = args.ext
gpu_num = args.gpu_num
if args.dtype == "float16":
    DTYPE = torch.float16
elif args.dtype == "float32":
    DTYPE = torch.float32
else:
    DTYPE = BASE_DTYPE

weight_path = args.weight_path
mask_threshold = args.mask_threshold
gpu_number = str(args.gpu_num)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_number
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'Target Directory: {target_dir}')
print(f'Output Directory: {output_dir}')
print(f'GPU Number: {gpu_num}')
print(f'Dtype: {DTYPE}')
print(f'Device: {device}')
print(f'Weight Path: {weight_path}')
print(f'Mask Threshold: {mask_threshold}')

if __name__ == '__main__':

    model_multi = InceptionResNetV2MultiTask3D(input_shape=(1, size, size, size),
                                            class_channel=2, seg_channels=num_classes, validity_shape=(1, 8, 8, 8),
                                            inject_class_channel=None,
                                            block_size=12, decode_init_channel=None,
                                            skip_connect=True, dropout_proba=0.05, norm="instance", act="relu6",
                                            class_act="softmax", seg_act="softmax", validity_act="sigmoid",
                                            get_seg=True, get_class=True, get_validity=False,
                                            use_class_head_simple=True, use_seg_pixelshuffle_only=False
                                            )
    model_multi = model_multi.eval().to(device=device, dtype=DTYPE)
    load_model(weight_path, model_multi)

    nii_path_list = glob(f"{target_dir}/{target_ext}/*.{target_ext}")
    center_point_df = pd.read_csv(f"{target_dir}/manual_pos.csv")
    point_dict = {}
    for index, row in center_point_df.iterrows():
        series_uid = row['SeriesInstanceUID']
        center_point = [row['x'], row['y'], row['z']]
        
        if series_uid in point_dict:
            point_dict[series_uid].append(center_point)
        else:
            point_dict[series_uid] = [center_point]
            
    valid_nii_path_list = [item for item in nii_path_list
                        if os.path.basename(item).replace(f".{target_ext}", "") in point_dict]

    use_stride = True
    resize = True
    patch_size = 96
    proj_name_list = ["axial", "coronal", "sagittal"]
    phase_list = ["jegal_seg_class"]
    model_list = [model_multi]

    with torch.no_grad():
        for use_stride in [True]:
            if use_stride == "1x1":
                result_folder = f"{output_dir}/96_single_cubic_{mask_threshold}"
            elif use_stride is True:
                result_folder = f"{output_dir}/stride_half_{mask_threshold}"
            else:
                result_folder = f"{output_dir}/non_stride_{mask_threshold}"
            for nii_path in tqdm(valid_nii_path_list):
                path_basename = os.path.basename(nii_path)
                if ".nii.gz" in path_basename:
                    series_uid = path_basename.replace(".nii.gz", "")
                else:
                    series_uid = path_basename.replace(".nii", "")
                
                result_visualize_folder = f"{result_folder}/visualize/{series_uid}"
                result_file_folder = f"{result_folder}/files"
                result_image_path = f"{result_file_folder}/{series_uid}_image.{target_ext}"
                result_mask_path = f"{result_file_folder}/{series_uid}_mask.{target_ext}"
                os.makedirs(result_visualize_folder, exist_ok=True)
                os.makedirs(result_file_folder, exist_ok=True)
                
                sitk_obj = sitk.ReadImage(nii_path)
                sitk_array = sitk.GetArrayFromImage(sitk_obj)
                slice_num = sitk_array.shape[0]
                ct_z, ct_h, ct_w = sitk_array.shape
                center_point_list = point_dict[series_uid]
                center_idx_list = [get_center_idx_from_point(sitk_obj, center_point)
                                for center_point in center_point_list]

                pred_array_list = []
                for target_model in model_list:
                    pred_array = np.zeros_like(sitk_array).astype("float32")
                    for center_idx in center_idx_list:
                        z_idx = center_idx[0]
                        if use_stride:
                            patch_batch, center_idx_part_list = get_5x5x5_patch_from_center_idx(sitk_array, center_idx, patch_size=patch_size)
                        elif use_stride == "1x1":
                            patch_batch, center_idx_part_list = get_1x1x1_patch_from_center_idx(sitk_array, center_idx, patch_size=patch_size)
                        else:
                            patch_batch, center_idx_part_list = get_3x3x3_patch_from_center_idx(sitk_array, center_idx, patch_size=patch_size)
                        patch_tensor = torch.tensor(preprocess(patch_batch)[:, None] / 255).float().to(device=device, dtype=DTYPE)
                        pred_patch_batch = process_patch_array(patch_tensor, target_model, threshold=mask_threshold).cpu().numpy()
                        for pred_patch_array, center_idx_part in zip(pred_patch_batch, center_idx_part_list):
                            pred_array = fill_value_in_mask(pred_array, pred_patch_array, center_idx_part, patch_size=patch_size)
                    pred_array_list.append(pred_array)
                for center_idx in center_idx_list:
                    z_idx, h_idx, w_idx = center_idx
                    # 사용자 정의 colormap 생성
                    colors = [(1, 1, 1, 0), (1, 0, 0)]  # 시작: 투명한 흰색, 끝: 빨간색
                    n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
                    cmap_name = 'custom_div_cmap'
                    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
                    fig, ax = plt.subplots(len(phase_list), 9, figsize=(72, 8*len(phase_list)))
                    for model_idx, phase in enumerate(phase_list):
                        pred_array = pred_array_list[model_idx]
                        for shape_idx, proj_name in enumerate(proj_name_list):
                            idx = center_idx[shape_idx]
                            num = sitk_array.shape[shape_idx]

                            center_point = deepcopy(center_idx)
                            center_point = list(center_point)
                            target_idx = center_point.pop(shape_idx)
                            x, y = center_point[1], center_point[0]
                            img_slice = np.take(sitk_array, indices=target_idx, axis=shape_idx).clip(CLIP_MIN, CLIP_MAX)
                            pred_slice = np.take(pred_array, indices=target_idx, axis=shape_idx)
                            if resize:
                                y = int(round(y * DATA_SIZE / img_slice.shape[0]))
                                img_slice = cv2.resize(img_slice, (DATA_SIZE, DATA_SIZE), interpolation=cv2.INTER_LINEAR_EXACT)
                                pred_slice = cv2.resize(pred_slice, (DATA_SIZE, DATA_SIZE), interpolation=cv2.INTER_NEAREST)

                            ax[shape_idx * 3].imshow(img_slice, cmap="gray")
                            ax[shape_idx * 3 + 1].imshow(img_slice, cmap="gray")
                            ax[shape_idx * 3 + 1].imshow(pred_slice, cmap=cm, alpha=0.25)
                            ax[shape_idx * 3 + 1].scatter(x, y, color='blue', marker='*')
                            ax[shape_idx * 3 + 2].imshow(pred_slice,cmap="gray")
                            ax[shape_idx * 3].set_title(f"{phase}_{proj_name}", fontsize=14)
                    plt.tight_layout()
                    visualize_save_path = f"{result_visualize_folder}/(z_{z_idx}_{ct_z})_(h_{h_idx}_{ct_h})_(w_{w_idx}_{ct_w}).png"
                    plt.savefig(visualize_save_path, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                save_numpy_to_nifti(sitk_array, result_image_path)
                save_numpy_to_nifti(pred_array, result_mask_path)
    