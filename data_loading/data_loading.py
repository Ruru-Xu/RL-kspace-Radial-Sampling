import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import nibabel as nib
import fastmri
import pathlib
from numpy.fft import fftshift, ifftshift, fftn, ifftn


# --------- Utility Functions ---------
def complex_abs(data):
    """Compute absolute value of a complex tensor"""
    assert data.shape[1] == 2  # Expecting (2, H, W) format for complex data
    return torch.sqrt((data ** 2).sum(axis=1))

def pad_and_center_crop(img, background_size=310, target_shape=(256, 256)):
    """
    先将 img 放到尺寸为 background_size x background_size 的背景中，
    若 img 尺寸超过背景则先做中心裁剪；之后再从背景中中心裁剪出 target_shape 大小的图像。
    """
    # 获取原始图像尺寸
    h, w = img.shape
    # 如果 img 大于背景尺寸，则先对 img 做中心裁剪至背景尺寸
    if h > background_size or w > background_size:
        start_y = (h - background_size) // 2 if h > background_size else 0
        start_x = (w - background_size) // 2 if w > background_size else 0
        img = img[start_y:start_y+min(h, background_size), start_x:start_x+min(w, background_size)]
        h, w = img.shape  # 更新尺寸

    # 创建背景（用0作为背景像素值，可根据需要调整）
    background = np.zeros((background_size, background_size), dtype=img.dtype)
    # 将 img 放置到背景的中心
    y_offset = (background_size - h) // 2
    x_offset = (background_size - w) // 2
    background[y_offset:y_offset+h, x_offset:x_offset+w] = img

    # 对背景再做中心裁剪至目标尺寸 target_shape
    crop_y = (background_size - target_shape[0]) // 2
    crop_x = (background_size - target_shape[1]) // 2
    cropped_img = background[crop_y:crop_y+target_shape[0], crop_x:crop_x+target_shape[1]]

    return cropped_img


# --------- Data Loading Class ---------
class MRIDataset(Dataset):
    """MRI Dataset for ACDC / fastMRI++ knee datasets, returns k-space data and corresponding image"""

    def __init__(self, data_path, dataset_type='ACDC', crop_size=(256, 256)):
        """
        Args:
            data_path (str): Path to dataset directory or HDF5 file.
            dataset_type (str): Type of dataset ('ACDC' or 'fastMRI_knee').
            crop_size (tuple): Target image size after cropping.
        """
        self.dataset_type = dataset_type
        self.crop_size = crop_size
        self.examples = []

        if dataset_type == 'ACDC':
            self.load_acdc(data_path)
        elif dataset_type == 'fastMRI_knee':
            self.load_fastmri_knee(data_path)
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def load_acdc(self, base_root):
        """Load ACDC dataset, extract k-space and segmentation masks"""
        patients_folder = sorted([f for f in base_root.iterdir() if f.is_dir()])
        for folder in patients_folder:
            for file in os.listdir(folder):
                if file.startswith(str(folder).split('/')[-1] + '_frame') and not file.endswith('gt.nii.gz'):
                    path_img = os.path.join(folder, file)
                    path_label = os.path.join(folder, file.split('.')[0] + '_gt.nii.gz')
                    data_label = nib.load(path_label).get_fdata()

                    for slice_idx in range(data_label.shape[-1]):
                        if np.any(data_label[:, :, slice_idx]):  # Only use slices with segmentation labels
                            self.examples.append((path_img, slice_idx, path_label))

    def load_fastmri_knee(self, h5_file):
        """Load fastMRI knee dataset from HDF5 file"""
        with h5py.File(h5_file, 'r') as hf:
            for i in range(len(hf['kspace'])):
                self.examples.append(i)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if self.dataset_type == 'ACDC':
            return self.get_acdc_sample(idx)
        elif self.dataset_type == 'fastMRI_knee':
            return self.get_fastmri_sample(idx)

    def get_acdc_sample(self, idx):
        """Load ACDC sample: k-space, image, and segmentation mask"""
        path_img, slice_idx, path_label = self.examples[idx]

        target_np = nib.load(path_img).get_fdata()[:, :, slice_idx].astype(np.float32)[:]
        # target_crop = pad_and_center_crop(target_np)
        target_norm = (target_np - target_np.min()) / (target_np.max() - target_np.min())

        label_np = nib.load(path_label).get_fdata()[:, :, slice_idx].astype(np.float32)[:]
        # seg_mask_crop = pad_and_center_crop(label_np)
        label_np[label_np != 0] = 1 # Set values to 1 where they are not equal to 0

        # 设置初始采样 mask（中心 10% 采样）
        initial_mask = np.zeros(target_norm.shape)
        center_fraction = 0.125
        center_size = int(target_norm.shape[-1] * center_fraction)
        start_idx = (target_norm.shape[-1] - center_size) // 2
        initial_mask[start_idx:start_idx + center_size, start_idx:start_idx + center_size] = 1

        kspace_fully = transform_image_to_kspace(target_norm, [0, 1])
        masked_kspace = kspace_fully * initial_mask
        undersampled_img = transform_kspace_to_image(masked_kspace, [0, 1])

        # return target_norm, kspace_fully, seg_mask_crop, masked_kspace, undersampled_img, initial_mask
        parameters = {
            "target": target_norm,
            "kspace_fully": kspace_fully,
            "seg_mask": label_np,
            "masked_kspace": masked_kspace,
            "undersampled_img": undersampled_img,
            'initial_mask': initial_mask,
        }
        return parameters

    def get_fastmri_sample(self, idx):
        """Load fastMRI knee sample: k-space and image"""
        with h5py.File(self.data_path, 'r') as hf:
            kspace_fully = hf['kspace'][idx]  # Shape: (H, W, 2)
            target = hf['target'][idx]  # Fully sampled image
            print('implement later')

def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions

    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions

    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = fftshift(fftn(ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


# --------- Data Loader Creation ---------
def create_dataloader(data_path, dataset_type, batch_size=8, num_workers=4, shuffle=True):
    """Create dataloader for MRI datasets"""
    dataset = MRIDataset(data_path, dataset_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)

if __name__ == '__main__':
    train_loader = create_dataloader(pathlib.Path("/home/ruru/Documents/work/ACDC/ACDC/database/testing"), dataset_type="ACDC", batch_size=8)
    for it, data in enumerate(train_loader):
        cbatch = 1
    print('ok')
    # train_loader = create_dataloader("path/to/fastmri_knee.h5", dataset_type="fastMRI_knee", batch_size=8)
