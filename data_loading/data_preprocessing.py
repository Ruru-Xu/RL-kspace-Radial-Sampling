import os
import pathlib

import torch
import numpy as np
import pandas as pd
import h5py
import tqdm
from typing import Tuple
import nibabel as nib

import fastmri
from fastmri.data import transforms as T


def center_crop_img(image, crop_size=256, pad_size=512):
    """
    将图像先拷贝到一个pad_size大小的全0背景中，然后再对该背景居中crop为crop_size大小。
    假设输入图像为 [H, W, Slices] 格式。
    返回最终裁剪后的图像，以及用于坐标偏移的参数 (start_h, start_w, final_start_h, final_start_w)。
    """
    # image = nib.load(file_name).get_fdata()
    H, W, S = image.shape  # H:高, W:宽, S:切片数

    # 创建一个pad_size x pad_size的全零背景，并保持相同的slice数
    background = np.zeros((pad_size, pad_size, S), dtype=image.dtype)

    # 将原图居中贴到background上
    start_h = (pad_size - H) // 2
    start_w = (pad_size - W) // 2
    background[start_h:start_h + H, start_w:start_w + W, :] = image

    # 在background上再居中crop到crop_size x crop_size
    final_start_h = (pad_size - crop_size) // 2
    final_start_w = (pad_size - crop_size) // 2
    final_image = background[final_start_h:final_start_h + crop_size, final_start_w:final_start_w + crop_size, :]

    # 返回图像
    return final_image


def norm_img(data_img):
    # 沿着(0,1)维度对每个slice归一化，即对每个slice独立求min和max
    # data_img.min(axis=(0,1)) 会返回一个大小为 [Slices] 的数组，表示每个slice的最小值
    # data_img.max(axis=(0,1)) 会返回一个大小为 [Slices] 的数组，表示每个slice的最大值

    slice_min = data_img.min(axis=(0, 1), keepdims=True)  # shape: (1,1,Slices)
    slice_max = data_img.max(axis=(0, 1), keepdims=True)  # shape: (1,1,Slices)

    # 防止出现除0错误
    denominator = slice_max - slice_min
    denominator[denominator == 0] = 1

    # 广播减法和除法会将 (H,W,Slices) 的数据与 (1,1,Slices) 的min和max匹配
    data_img_norm = (data_img - slice_min) / denominator  # shape仍为 (H,W,Slices)，范围在[0,1]
    return data_img_norm


def img2kspace(data_img_norm):
    img_tensor = torch.tensor(data_img_norm).float().permute(2, 0, 1)
    img_tensor = torch.stack((img_tensor, torch.zeros_like(img_tensor)), dim=-1)  # [1,1,H,W,2]
    kspace = fastmri.fft2c(img_tensor)
    return kspace


def process_file(data_paths, mean, Rrr, Rii, Rri, output_key="kspace_scaled"):
    # Load and process k-space data for a single file
    patients_folder = sorted([f for f in data_paths.iterdir() if f.is_dir()])
    for folder in tqdm.tqdm(patients_folder, desc="Computing mean and covariance"):
        for file in os.listdir(folder):
            if file.startswith(str(folder).split('/')[-1] + '_frame') and not file.endswith('gt.nii.gz'):
                path_img = os.path.join(folder, file)
                img_nib = nib.load(path_img)  # Keep the original NiBabel object for header info
                data_img_crop_norm = img_nib.get_fdata()[:]
                kspace = img2kspace(data_img_crop_norm)

                data_fft = fastmri.ifft2c(kspace)
                data_fft = torch.complex(data_fft[..., 0], data_fft[..., 1]).type(torch.complex64)

                # Scale data
                slice_out = data_fft - mean
                scaled_data = (Rrr[None, None] * slice_out.real + Rri[None, None] * slice_out.imag).type(torch.complex64) \
                      + 1j * (Rii[None, None] * slice_out.imag + Rri[None, None] * slice_out.real).type(torch.complex64)

                out_img = torch.abs(scaled_data.permute(1, 2, 0)).numpy()
                new_img = nib.Nifti1Image(out_img, img_nib.affine, img_nib.header)
                nib.save(new_img, path_img)



def compute_mean_cov(data_paths):
    # Compute mean and covariance
    mean_real, mean_imag, numel = 0, 0, 0
    sum_real, sum_imag = 0, 0

    patients_folder = sorted([f for f in data_paths.iterdir() if f.is_dir()])
    for folder in tqdm.tqdm(patients_folder, desc="Computing mean and covariance"):
        for file in os.listdir(folder):
            if file.startswith(str(folder).split('/')[-1] + '_frame') and not file.endswith('gt.nii.gz'):
                path_img = os.path.join(folder, file)
                # Load original data
                img_nib = nib.load(path_img)  # Keep the original NiBabel object for header info
                data_img_crop_norm = img_nib.get_fdata()[:]
                kspace = img2kspace(data_img_crop_norm)

                data_fft = fastmri.ifft2c(kspace)
                data_fft = torch.complex(data_fft[..., 0], data_fft[..., 1]).type(torch.complex64)

                sum_real += data_fft.real.sum()
                sum_imag += data_fft.imag.sum()
                numel += data_fft.numel()

    # Calculate mean
    mean_real = sum_real / numel
    mean_imag = sum_imag / numel
    mean = torch.complex(mean_real, mean_imag)

    # Compute covariance
    Crr, Cii, Cri = 0, 0, 0
    for folder in tqdm.tqdm(patients_folder, desc="Computing mean and covariance"):
        for file in os.listdir(folder):
            if file.startswith(str(folder).split('/')[-1] + '_frame') and not file.endswith('gt.nii.gz'):
                path_img = os.path.join(folder, file)
                data_img_crop_norm = nib.load(path_img).get_fdata()[:]
                kspace = img2kspace(data_img_crop_norm)

                data_fft = fastmri.ifft2c(kspace)
                data_fft = torch.complex(data_fft[..., 0], data_fft[..., 1]).type(torch.complex64)

                centered_data = data_fft - mean
                Crr += centered_data.real.pow(2).sum()
                Cii += centered_data.imag.pow(2).sum()
                Cri += (centered_data.real * centered_data.imag).sum()

    # Normalize covariance
    Crr /= numel
    Cii /= numel
    Cri /= numel
    eps = 1e-8  # Avoid division by zero

    # Calculate the inverse square root of the covariance matrix
    det = Crr * Cii - Cri.pow(2)
    s = torch.sqrt(det + eps)
    t = torch.sqrt(Cii + Crr + 2 * s)
    inverse_st = 1.0 / (s * t + eps)
    Rrr = (Cii + s) * inverse_st
    Rii = (Crr + s) * inverse_st
    Rri = -Cri * inverse_st

    return mean, Rrr, Rii, Rri

def save_crop(data_paths):
    patients_folder = sorted([f for f in data_paths.iterdir() if f.is_dir()])
    for folder in tqdm.tqdm(patients_folder, desc="Computing mean and covariance"):
        for file in os.listdir(folder):
            if file.startswith(str(folder).split('/')[-1] + '_frame') and not file.endswith('gt.nii.gz'):
                path_img = os.path.join(folder, file)
                path_label = os.path.join(folder, file.split('.')[0] + '_gt.nii.gz')

                # Load original data
                img_nib = nib.load(path_img)  # Keep the original NiBabel object for header info
                data_img = img_nib.get_fdata()[:]
                data_label = nib.load(path_label).get_fdata()[:]

                data_img_crop = center_crop_img(data_img)
                data_label_crop = center_crop_img(data_label)
                data_img_crop_norm = np.float32(norm_img(data_img_crop))

                save_folder = str(folder).replace('database', 'spiral_data')
                # Create directory if it doesn't exist
                os.makedirs(save_folder, exist_ok=True)
                output_img_path = os.path.join(save_folder, file)
                output_label_path = os.path.join(save_folder, file.split('.')[0] + '_gt.nii.gz')

                # Create new NIfTI images with original affine and header
                new_img = nib.Nifti1Image(data_img_crop_norm, img_nib.affine, img_nib.header)
                new_label = nib.Nifti1Image(data_label_crop, img_nib.affine, img_nib.header)

                nib.save(new_img, output_img_path)
                nib.save(new_label, output_label_path)

                print(f"Saved processed image to: {output_img_path}")
                print(f"Saved processed label to: {output_label_path}")

if __name__ == '__main__':
    # train_path = pathlib.Path("/home/ruru/Documents/work/ACDC/ACDC/database/test")
    # save_crop(train_path)

    train_path = pathlib.Path("/home/ruru/Documents/work/ACDC/ACDC/spiral_data/train")
    val_path = pathlib.Path("/home/ruru/Documents/work/ACDC/ACDC/spiral_data/val")
    test_path = pathlib.Path("/home/ruru/Documents/work/ACDC/ACDC/spiral_data/test")

    # Compute mean and covariance from training data
    mean, Rrr, Rii, Rri = compute_mean_cov(train_path)

    # Process training, validation, and test sets
    print("Processing training data...")
    process_file(train_path, mean, Rrr, Rii, Rri)

    print("Processing val data...")
    process_file(val_path, mean, Rrr, Rii, Rri)

    print("Processing test data...")
    process_file(test_path, mean, Rrr, Rii, Rri)
