import sys
import os
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from glob import glob
import ast
import cv2
from matplotlib import pyplot as plt
from src.data_set.segmentation import SegDataset
from src.data_set.utils import read_json_as_dict
import random
import pandas as pd
import SimpleITK as sitk
from torch import nn
from natsort import natsorted
from utils import read_dicom_series_to_array, get_dicom_series_shape, read_nii_to_array, get_nii_shape
from utils import resize_dicom_series, write_series_to_path
from utils import get_parent_dir_name
from volumentations import Compose, ElasticTransform, RandomGamma, GaussianNoise, Transpose, Flip, RandomRotate90, GlassBlur, RandomCrop, GridDropout

from itertools import chain
import deepspeed
import math
from copy import deepcopy
from src.loss.seg_loss import get_dice_score, get_loss_fn, get_bce_loss_class, accuracy_metric
from src.model.inception_resnet_v2.multi_task.multi_task_3d import InceptionResNetV2MultiTask3D
from src.util.deepspeed import get_deepspeed_config_dict, average_across_gpus, toggle_grad, load_deepspeed_model_to_torch_model
from src.util.common import set_dropout_probability
import torch.nn.functional as F
from src.model.inception_resnet_v2.multi_task.multi_task_3d import InceptionResNetV2MultiTask3D
from src.model.train_util.logger import CSVLogger
import torch.distributed as dist
import pandas as pd
import csv
import nibabel as nib

def get_add_splited_fold_column_df(source_df, n_fold=10):
    target_df = source_df.copy()
    fold_idx_list = list(range(len(target_df)))
    fold_idx_splited = [None for _ in fold_idx_list]

    np.random.shuffle(fold_idx_list)
    fold_idx_list_split = np.array_split(fold_idx_list, n_fold)

    for split_idx, fold_idx_split in enumerate(fold_idx_list_split):
        for fold_idx in fold_idx_split:
            fold_idx_splited[fold_idx] = split_idx

    target_df["Fold"] = fold_idx_splited
    return target_df

def compute_loss_metric(model, x, y, y_label, get_class, get_recon):
    ######## Compute Loss ########
    model_output = model(x)
    if get_class and get_recon:
        y_pred, y_label_pred, y_recon_pred = model_output
        seg_loss = get_loss(y_pred, y)
        class_loss = get_class_loss(y_label_pred, y_label)
        recon_loss = get_recon_loss(y_recon_pred, x, y_pred)
        loss = seg_loss + class_loss + recon_loss
    elif get_class:
        y_pred, y_label_pred = model_output
        class_loss = get_class_loss(y_label_pred, y_label)
        seg_loss = get_loss(y_pred, y)
        loss = seg_loss + class_loss.m
    elif get_recon:
        y_pred, y_recon_pred = model_output
        seg_loss = get_loss(y_pred, y)
        recon_loss = get_recon_loss(y_recon_pred, x, y_pred)
        loss = seg_loss + recon_loss
    else:
        y_pred = model_output
        seg_loss = get_loss(y_pred, y)
        loss = seg_loss
    ######## Compute Metric #######
    with torch.no_grad():
        _, y_pred = torch.max(y_pred, dim=1)
        y_pred = F.one_hot(y_pred, num_classes=num_classes)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        metric_dict = {
            "loss": average_across_gpus(loss),
            "dice_score": average_across_gpus(get_dice_score(y_pred, y))
        }
        if get_class:
            metric_dict["accuracy"] = average_across_gpus(accuracy_metric(y_label_pred, y_label))
        if get_recon:
            metric_dict["recon_diff"] = average_across_gpus(recon_loss)
    return loss, metric_dict

def update_loss_score_dict(loss_score_dict, get_class, get_recon, metric_dict):
    if metric_dict is not None:
        loss_score_dict["loss_list"].append(metric_dict["loss"])
        loss_score_dict["dice_score_list"].append(metric_dict["dice_score"])
        if get_class:
            loss_score_dict["accuracy_list"].append(metric_dict["accuracy"])
        if get_recon:
            loss_score_dict["recon_diff_list"].append(metric_dict["recon_diff"])

def print_and_save_log(train_loss_score_dict, val_loss_score_dict, csv_logger, epoch, get_class, get_recon):
    if dist.get_rank() == 0:
        data_info_str = [f'{epoch}']
        train_info_str = [f'{np.mean(train_loss_score_dict["loss_list"]):.4f}',
                        f'{np.mean(train_loss_score_dict["dice_score_list"]):.4f}']
        val_info_str = [f'{np.mean(val_loss_score_dict["dice_score_list"]):.4f}']

        if get_class:
            train_info_str.append(f'{np.mean(train_loss_score_dict["accuracy_list"]):.4f}')
        if get_recon:
            train_info_str.append(f'{np.mean(train_loss_score_dict["recon_diff_list"]):.4f}')

        data_info_str = data_info_str + train_info_str + val_info_str
        csv_logger.writerow([*data_info_str])
        data_info_str = " - ".join(data_info_str)
        print(data_info_str)

def get_loss_score_dict():
    return {
        "loss_list": [],
        "dice_score_list": [],
        "accuracy_list": [],
        "recon_diff_list": []
    }
def get_loss_score_dict_full():
    return {
        "dice_score_list": [],
    }


def get_processed_data(x, y, device, dtype):
    x = x.to(device=device, dtype=dtype)
    y = y.to(device=device, dtype=torch.long)
    y_label = F.one_hot((y.sum(dim=[1, 2, 3]) > 0).long(), num_classes)
    y = F.one_hot(y, num_classes).permute(0, 4, 1, 2, 3).to(dtype=dtype)
    return x, y, y_label

def full_data_collate_fn(batch):
    # shape 이 달라 list로 따로 들어있다
    image_tensor_list, mask_tensor_list = zip(*batch)
    return image_tensor_list, mask_tensor_list

def update_loss_score_dict_full(loss_score_dict, dice_score):
    if metric_dict is not None:
        loss_score_dict["dice_score_list"].append(dice_score)

def compute_loss_metric_full(model, x_list, y_list, target_z_dim, process_batch_size,
                             get_class, get_recon, use_class_in_predict, device, dtype):
    
    y_pred_list, dice_score_list = model_predict_full(model, x_list, y_list, target_z_dim, process_batch_size,
                                                      get_class, get_recon, use_class_in_predict, device, dtype)
    return y_pred_list, dice_score_list

def model_predict_full(model, x_list, y_list, target_z_dim, process_batch_size,
                  get_class, get_recon, use_class_in_predict, device, dtype):
    with torch.no_grad():
        y_pred_list = []
        dice_score_list_per_gpu = []
        for batch_x, batch_y in zip(x_list, y_list):
            batch_x, batch_y = batch_x[None], batch_y[None]
            stride = target_z_dim // 4
            batch_y_pred = torch.zeros_like(batch_y)
            batch_info_list = []
            z_dim = batch_x.shape[2]
            z_idx_range = range(0, z_dim - target_z_dim + stride, stride)
            for idx, z_idx in enumerate(z_idx_range):
                x_slice = batch_x[:, :, z_idx:z_idx+target_z_dim]

                pad_num = target_z_dim - x_slice.shape[2]
                pad_half = pad_num // 2
                x_slice = F.pad(x_slice, (0, 0, 0, 0, pad_half, pad_num - pad_half), "constant", 0)
                batch_info_list.append([z_idx, pad_num, x_slice])

                if len(batch_info_list) == process_batch_size or idx < len(z_idx_range):
                    slice_batch = [batch_info[-1] for batch_info in batch_info_list]
                    slice_batch = torch.cat(slice_batch, dim=0).to(device=device, dtype=dtype)

                    if get_class and get_recon:
                        batch_predict, batch_label_predict, batch_recon_predict = model(slice_batch)
                    elif get_class:
                        batch_predict, batch_label_predict = model(slice_batch)
                    elif get_recon:
                        batch_predict, batch_recon_predict = model(slice_batch)
                    else:
                        batch_predict = model(slice_batch)

                    if get_class and use_class_in_predict:
                        # batch_indices.shape = [B, C]
                        # batch_predict.shape = [B, C, D, H, W]
                        batch_predict = batch_predict * batch_label_predict[:, :, None, None, None]

                    for idx, (slice_z_idx, pad_num) in enumerate([batch_info[:2] for batch_info in batch_info_list]):
                        pad_half = pad_num // 2
                        y_slice_pred = batch_predict[idx][None]
                        start_idx = slice_z_idx
                        end_idx = min(slice_z_idx + target_z_dim, z_dim)
                        z_slice = slice(start_idx, end_idx)
                        # total - pad_num => total - pad_num + pad_half
                        part_z_slice = slice(pad_half, end_idx - slice_z_idx + pad_half)
                        previous_slice = batch_y_pred[:, z_slice]
                        current_slice = y_slice_pred[:, :, part_z_slice].argmax(1).cpu()
                        batch_y_pred[:, z_slice] = torch.maximum(previous_slice, current_slice)
                    batch_info_list = []
            y_pred_list.append(batch_y_pred)

        for y_pred, y in zip(y_pred_list, y_list):
            epsilon = 1e-7
            y_pred, y = y_pred[0].numpy(), y[0].numpy()
            tp = np.sum(y_pred * y)
            fp = np.sum(y_pred) - tp
            fn = np.sum(y) - tp
            dice_score = (2 * tp + epsilon) / (2 * tp + fp + fn + epsilon)
            dice_score = torch.tensor(dice_score).to(device=device, dtype=dtype)
            dice_score_list_per_gpu.append(dice_score)
        dice_score_per_gpu = torch.stack(dice_score_list_per_gpu, dim=0).mean()
        dice_score_reduced = average_across_gpus(dice_score_per_gpu)
    
    return y_pred_list, dice_score_reduced

n_fold = 10

target_z_dim = 32
loss_list = ["dice_bce", "dice_bce_focal", "tversky_bce", "propotional_bce"]

total_epoch = 10
stage_coef_list = [2, 5]
decay_epoch = total_epoch - sum(stage_coef_list)
decay_dropout_ratio = 0.25 ** (1  / (total_epoch - sum(stage_coef_list)))
lr_setting_list = [4e-5, 2e-4, 0.25]

test_model = "unet_custom"
loss_select = "propotional_bce"
in_channels = 1
num_classes = 2
batch_size = 8
get_class = True
get_recon = True
use_seg_in_recon = True

num_gpu = torch.cuda.device_count()
loader_batch_size = 2 * num_gpu
num_workers = min(loader_batch_size * 2, 16)

def apply_augmentation(transform, image_array, mask_array):
    transform_dict = transform(image=image_array, mask=mask_array)
    
    return transform_dict["image"], transform_dict["mask"]

def get_augmentation():
    return Compose([
#         GridDropout(0.5, fill_value=0, mask_fill_value=0, p=1.0),
#         ElasticTransform(deformation_limits=(0, 0.15), p=1.0),
#         GlassBlur(sigma=0.05, max_delta=2, iterations=2, always_apply=False, mode='fast', p=0.5),
#         RandomGamma(gamma_limit=(80, 120), p=0.35),
        GaussianNoise(var_limit=(0, 5), p=0.5),
#         Transpose(p=0.5),
        Flip(0, p=0.5),
        Flip(1, p=0.5),
        Flip(2, p=0.5),
        RandomRotate90((1, 2), p=0.5),
    ], p=1.0)

def read_image(image_path):
    image_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image_array

def image_preprocess(image_array):
    min_value, max_value = 0, 255
    image_array = image_array.clip(min_value, max_value)
    image_array = (image_array - min_value) / (max_value - min_value)
    image_array = image_array / 255
    return image_array.astype("float32")[None]

def mask_preprocess(mask_array):
    mask_array = mask_array / 255
    return mask_array.astype("float32")

def get_partial_3d_model(target_z_dim, num_classes, get_class, get_recon):
    return InceptionResNetV2MultiTask3D(input_shape=(1, target_z_dim, 512, 512),
                                        class_channel=num_classes, seg_channels=num_classes, validity_shape=(1, 8, 8, 8),
                                        inject_class_channel=None,
                                        block_size=6, decode_init_channel=None,
                                        skip_connect=True, dropout_proba=0.05, norm="instance", act="relu6",
                                        class_act="softmax", seg_act="softmax", validity_act="sigmoid",
                                        get_seg=True, get_class=get_class, get_recon=get_recon, get_validity=False,
                                        use_class_head_simple=True, use_decode_pixelshuffle_only=False, use_decode_simpleoutput=False
                                        )
def get_log_path(test_model, loss_select, batch_size, target_z_dim, current_fold):
    log_path = f"./result/3d_{test_model}_{loss_select}_{batch_size}"
    log_path = f"{log_path}_{target_z_dim}"
    if get_class:
        log_path = f"{log_path}_class"
    if get_recon:
        log_path = f"{log_path}_recon"
    log_path = f"{log_path}_fold_{current_fold}"
    os.makedirs(f"{log_path}/weights", exist_ok=True)
    return log_path

def get_best_dice_epoch(csv_path, select_mode="dice_score"):

    select_mode_list = ["loss", "dice_score", "dice_score_diff"]
    assert select_mode in select_mode_list, f"check your select_mode: {select_mode} in {select_mode_list}"

    with open(csv_path) as csv_file:
        reader = csv.DictReader(csv_file)
        dict_from_csv = {field_name:[] for field_name in reader.fieldnames}    
        for row in reader:
            for filedname in reader.fieldnames:
                dict_from_csv[filedname].append(float(row[filedname]))
    if select_mode == "loss":
        loss_min_epoch = np.argmin(dict_from_csv['val_loss']) + 1
        loss_min_loss = np.min(dict_from_csv['val_loss'])
        loss_min_score = dict_from_csv['val_dice_score'][loss_min_epoch - 1]
        return loss_min_epoch, loss_min_loss, loss_min_score
    elif select_mode == "dice_score":
        score_max_epoch = np.argmax(dict_from_csv['val_dice_score']) + 1
        score_max_loss = dict_from_csv['val_loss'][score_max_epoch - 1]
        score_max_score = np.max(dict_from_csv['val_dice_score'])
        return score_max_epoch, score_max_loss, score_max_score

    elif select_mode == "dice_score_diff":
        min_epoch = 5
        val_score = dict_from_csv['val_dice_score'][min_epoch:]
        score_diff = np.array(dict_from_csv['dice_score'] - np.array(dict_from_csv['val_dice_score']))[min_epoch:]
        score_diff = np.maximum(score_diff, 0)

        loss_score_diff_min_epoch = np.argmax(val_score - score_diff) + 1 + min_epoch
        loss_score_diff_min_loss = dict_from_csv['val_loss'][loss_score_diff_min_epoch - 1]
        loss_score_diff_min_score = dict_from_csv['val_dice_score'][loss_score_diff_min_epoch - 1]
        return loss_score_diff_min_epoch, loss_score_diff_min_loss, loss_score_diff_min_score
    
class SegDataset(Dataset):
    def __init__(self, data_folder_list, z_dim_list, target_z_dim, use_full):
        self.data_folder_list = data_folder_list
        self.z_dim_list = z_dim_list
        
        self.target_z_dim = target_z_dim
        self.use_full = use_full
        self.transform = get_augmentation()
        
    def __len__(self):
        return len(self.data_folder_list)
    
    def __getitem__(self, idx):
        data_folder, z_dim = self.data_folder_list[idx], self.z_dim_list[idx]
        image_path = f"{data_folder}/nii.gz/image.nii.gz"
        mask_path = f"{data_folder}/nii.gz/mask.nii.gz"
        image_array, mask_array = self.get_array_from_folder(image_path, mask_path, z_dim)
        image_array, mask_array = apply_augmentation(self.transform, image_array, mask_array)
        image_array = image_preprocess(image_array)
        mask_array = mask_preprocess(mask_array)
        
        return torch.tensor(image_array), torch.tensor(mask_array)
    
    def get_array_from_folder(self, image_path, mask_path, z_dim):
        target_z_dim = self.target_z_dim
        
        if target_z_dim > z_dim:
            z_idx = 0
            padding_num = target_z_dim - z_dim
            z_idx_range = range(z_idx, min(z_idx + target_z_dim, z_dim))
        else:
            if self.use_full:
                z_idx = 0
                padding_num = 0
                z_idx_range = range(z_idx, z_dim)
            else:
                cand_z_idx_list = [idx for idx in range(0, z_dim - target_z_dim)]
                z_idx = random.choice(cand_z_idx_list)
                padding_num = z_dim - z_idx if z_dim - z_idx < target_z_dim else 0
                z_idx_range = range(z_idx, min(z_idx + target_z_dim, z_dim))
        top_pad_num = padding_num // 2
        bottom_pad_num = padding_num - top_pad_num
        
        image_array = self.read_nii_gz(image_path)
        mask_array = self.read_nii_gz(mask_path)

        image_array = image_array[list(z_idx_range)]
        mask_array = mask_array[list(z_idx_range)]
        image_array = np.pad(image_array, [(top_pad_num, bottom_pad_num), (0, 0), (0, 0)],
                             mode="constant", constant_values=0)
        mask_array = np.pad(mask_array, [(top_pad_num, bottom_pad_num), (0, 0), (0, 0)],
                             mode="constant", constant_values=0)
        return image_array, mask_array
    
    def read_nii_gz(self, nii_path):
        image_obj = nib.load(nii_path)
        image_array = image_obj.get_fdata()
        image_array = image_array.transpose(2, 0, 1)
        return image_array
    
# def model_predict(model, batch_x, batch_y, target_z_dim, process_batch_size,
#                   get_class, get_recon, use_class_in_predict, device):
    
#     stride = target_z_dim // 4
#     batch_y_pred = torch.zeros_like(batch_y)
#     batch_info_list = []
#     z_dim = batch_x.shape[2]
#     z_idx_range = range(0, z_dim - target_z_dim + stride, stride)
#     for idx, z_idx in enumerate(z_idx_range):
#         x_slice = batch_x[:, :, z_idx:z_idx+target_z_dim]

#         pad_num = target_z_dim - x_slice.shape[2]
#         pad_half = pad_num // 2
#         x_slice = F.pad(x_slice, (0, 0, 0, 0, pad_half, pad_num - pad_half), "constant", 0)
#         batch_info_list.append([z_idx, pad_num, x_slice])
        
#         if len(batch_info_list) == process_batch_size or idx < len(z_idx_range):
#             slice_batch = [batch_info[-1] for batch_info in batch_info_list]
#             slice_batch = torch.cat(slice_batch, dim=0).to(device=device)
            
#             if get_class and get_recon:
#                 batch_predict, batch_label_predict, batch_recon_predict = model(slice_batch)
#             elif get_class:
#                 batch_predict, batch_label_predict = model(slice_batch)
#             elif get_recon:
#                 batch_predict, batch_recon_predict = model(slice_batch)
#             else:
#                 batch_predict = model(slice_batch)
            
#             if get_class and use_class_in_predict:
#                 # batch_indices.shape = [B, C]
#                 # batch_predict.shape = [B, C, D, H, W]
#                 batch_predict = batch_predict * batch_label_predict[:, :, None, None, None]

#             for idx, (slice_z_idx, pad_num) in enumerate([batch_info[:2] for batch_info in batch_info_list]):
#                 pad_half = pad_num // 2
#                 y_slice_pred = batch_predict[idx][None]
#                 start_idx = slice_z_idx
#                 end_idx = min(slice_z_idx + target_z_dim, z_dim)
#                 z_slice = slice(start_idx, end_idx)
#                 # total - pad_num => total - pad_num + pad_half
#                 part_z_slice = slice(pad_half, end_idx - slice_z_idx + pad_half)
#                 previous_slice = batch_y_pred[:, z_slice]
#                 current_slice = y_slice_pred[:, 1, part_z_slice].cpu()
#                 batch_y_pred[:, z_slice] = torch.maximum(previous_slice, current_slice)
#             batch_info_list = []
#     return batch_y_pred

meta_df_path = f"./phase.csv"
for current_fold in range(n_fold):
    train_valid_fold = list(range(0, n_fold))
    test_fold = current_fold
    del train_valid_fold[test_fold]

    meta_df = pd.read_csv(meta_df_path)
    meta_df_base, meta_df_ext = os.path.splitext(meta_df_path)
    meta_fold_df_path = f"{meta_df_base}_fold{meta_df_ext}"
    if os.path.exists(meta_fold_df_path):
        meta_fold_df = pd.read_csv(meta_fold_df_path)
    else:
        meta_fold_df = get_add_splited_fold_column_df(meta_df)
        meta_fold_df.to_csv(meta_fold_df_path, index=False)

    train_valid_folder_list = list(meta_fold_df[meta_fold_df["Fold"].isin(train_valid_fold)]["data_folder"])
    train_valid_z_dim_list = list(meta_fold_df[meta_fold_df["Fold"].isin(train_valid_fold)]["Depth"])
    test_folder_list = list(meta_fold_df[meta_fold_df["Fold"] == test_fold]["data_folder"])
    test_z_dim_list = list(meta_fold_df[meta_fold_df["Fold"] == test_fold]["Depth"])

    test_num = len(test_folder_list)
    valid_num = test_num
    train_valid_idx_list = list(range(len(train_valid_folder_list)))
    random.shuffle(train_valid_idx_list)

    train_idx_list, valid_idx_list = train_valid_idx_list[:-valid_num], train_valid_idx_list[-valid_num:]
    train_idx_list, valid_idx_list = sorted(train_idx_list), sorted(valid_idx_list)
    train_folder_list = [train_valid_folder_list[idx] for idx in train_idx_list]
    train_z_dim_list = [train_valid_z_dim_list[idx] for idx in train_idx_list]
    valid_folder_list = [train_valid_folder_list[idx] for idx in valid_idx_list]
    valid_z_dim_list = [train_valid_z_dim_list[idx] for idx in valid_idx_list]

    train_dataset = SegDataset(train_folder_list, train_z_dim_list, target_z_dim, use_full=False)
    val_dataset = SegDataset(valid_folder_list, valid_z_dim_list, target_z_dim, use_full=True)
    test_dataset = SegDataset(test_folder_list, test_z_dim_list, target_z_dim, use_full=True)
    print(f"train_valid_test: {len(train_dataset)}_{len(val_dataset)}_{len(test_dataset)}")

    base_model = get_partial_3d_model(target_z_dim, num_classes, get_class, get_recon)
    model_param_num = sum(p.numel() for p in base_model.parameters())
    print(f"model_param_num = {model_param_num}")

    get_l1_loss = nn.L1Loss()
    get_l2_loss = nn.MSELoss()
    # y_recon_pred.shape = [B, C, H, W]
    def get_recon_loss_follow_seg(y_recon_pred, y_recon_gt, y_seg_pred):
        img_dim = y_recon_pred.dim() - 2
        repeat_tuple = (1 for _ in range(img_dim))
        recon_image_channel = y_recon_pred.size(1)
        y_seg_pred_weight = 2 * torch.sigmoid(25 * y_seg_pred[:, 1]) - 1
        y_seg_pred_weight = y_seg_pred_weight.unsqueeze(1).repeat(1, recon_image_channel, *repeat_tuple)
        recon_loss = torch.abs(y_recon_pred - y_recon_gt) * y_seg_pred_weight
        return torch.mean(recon_loss)

    if use_seg_in_recon:
        get_recon_loss = get_recon_loss_follow_seg
    else:
        get_recon_loss = lambda y_recon_pred, y_recon_gt, _: get_l1_loss(y_recon_pred, y_recon_gt)
    get_class_loss = get_bce_loss_class
    get_loss = get_loss_fn(loss_select)
    
    ds_config = get_deepspeed_config_dict(train_dataset, loader_batch_size, batch_size, num_workers,
                                        stage_coef_list=stage_coef_list, decay_epoch=decay_epoch,
                                        cycle_min_lr=lr_setting_list[0], cycle_max_lr=lr_setting_list[1], decay_lr_rate=lr_setting_list[2])
    train_ds_config = ds_config
    val_ds_config = deepcopy(ds_config)

    _, _, val_loader, _ = deepspeed.initialize(
        model=base_model,
        model_parameters=base_model.parameters(),
        config=val_ds_config,
        training_data=val_dataset,
        collate_fn=full_data_collate_fn
    )
    _, _, test_loader, _ = deepspeed.initialize(
        model=base_model,
        model_parameters=base_model.parameters(),
        config=val_ds_config,
        training_data=test_dataset,
        collate_fn=full_data_collate_fn
    )
    model, _, train_loader, _ = deepspeed.initialize(
        model=base_model,
        model_parameters=base_model.parameters(),
        config=train_ds_config,
        training_data=train_dataset
    )

    device = model.device
    dtype = next(model.parameters()).dtype
    print(dtype)

    local_rank = dist.get_rank()

    if local_rank == 0:
        log_path = get_log_path(test_model, loss_select, batch_size, target_z_dim, current_fold)

        epoch_col = ["epoch"]
        train_col = ["loss", "dice_score"]
        val_col = ["val_dice_score"]
        if get_class:
            train_col.append("accuracy")
        if get_recon:
            train_col.append("max_recon_diff")

        csv_logger = CSVLogger(f"{log_path}/log.csv", epoch_col + train_col + val_col)
    else:
        csv_logger = None
        log_path = None

    if local_rank == 0:
        pbar_fn = tqdm
    else:
        pbar_fn = lambda x: x
    for epoch in range(1, total_epoch + 1):
        train_pbar = pbar_fn(train_loader)
        toggle_grad(model, require_grads=True)
        model.train()
        train_loss_score_dict = get_loss_score_dict()
        val_loss_score_dict = get_loss_score_dict_full()
        for batch_idx, (x, y) in enumerate(train_pbar, start=1):
            model.zero_grad()
            x, y, y_label = get_processed_data(x, y, device, dtype)
            loss, metric_dict = compute_loss_metric(model, x, y, y_label, get_class, get_recon)
            update_loss_score_dict(train_loss_score_dict, get_class, get_recon, metric_dict)
            model.backward(loss)
            model.step()
            if local_rank == 0:
                train_pbar.set_postfix({'Epoch': f'{epoch}/{total_epoch}',
                                        'loss': f'{np.mean(train_loss_score_dict["loss_list"]):.4f}',
                                        'dice_score': f'{np.mean(train_loss_score_dict["dice_score_list"]):.4f}',
                                        'current_loss':f'{train_loss_score_dict["loss_list"][-1]:.4f}'})
        toggle_grad(model, require_grads=False)

        model.eval()
        with torch.no_grad():
            valid_pbar = pbar_fn(val_loader)
            for batch_idx, (x_list, y_list) in enumerate(valid_pbar):
                y_pred_list, dice_score = compute_loss_metric_full(model, x_list, y_list, target_z_dim, process_batch_size=batch_size, 
                                                                   get_class=get_class, get_recon=get_recon, use_class_in_predict=False, 
                                                                   device=device, dtype=dtype)
                update_loss_score_dict_full(val_loss_score_dict, dice_score)
        set_dropout_probability(model, decay_dropout_ratio=decay_dropout_ratio)
        print_and_save_log(train_loss_score_dict, val_loss_score_dict, csv_logger, epoch, get_class, get_recon)
        if local_rank == 0:
            torch.save(model.state_dict(), f"{log_path}/weights/{epoch:03}.ckpt")
    
# for current_fold in range(n_fold):
#     base_model = get_partial_3d_model(target_z_dim, num_classes, get_class, get_recon)
#     log_path = get_log_path(test_model, loss_select, batch_size, target_z_dim, current_fold)
#     log_csv_path = f"{log_path}/log.csv"
#     best_epoch, best_loss, best_dice_score = get_best_dice_epoch(log_csv_path, select_mode="dice_score")
#     best_epoch_weight_path = f"{log_path}/{best_epoch:03}.ckpt"
#     load_deepspeed_model_to_torch_model(infer_model, best_epoch_weight_path)