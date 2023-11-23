import os
from glob import glob
import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image
import multiprocessing
from tqdm import tqdm
from functools import partial

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import Grayscale

from batchgenerators.augmentations.color_augmentations import augment_contrast
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.tuanluc_dev.utils import *
import monai.transforms as mt
import SimpleITK as sitk
import cv2

from batchgenerators.transforms.color_transforms import ContrastAugmentationTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
import nibabel as nib

"""
The default BraTS dataloader is for BraTS2018
"""
    
def get_BRATSDataset_dataloader(root_dir, batch_size, num_workers):
    
    train_transform = mt.Compose(
        [
            mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
        ]
    )
    val_transform = mt.Compose(
        [
            mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
        ]
    )
    
    train_dataset = BRATSDataset(root_dir, train=True, train_transform=train_transform, fold=0)
    val_dataset = BRATSDataset(root_dir, train=False, val_transform=val_transform, fold=0)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    sampler = StratifiedBatchSampler(train_dataset.labels, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


def slice_brats_utils(npy, filename, resize_transform, save_dir, save=False):
    os.makedirs(save_dir, exist_ok=True)
    npy = resize_transform(npy).numpy()
    map_modalities_dict = {0: "t1", 1: "t1ce", 2: "t2", 3: "flair"} # who knows the order?
    map_plans_dict = {0: "axial", 1: "coronal", 2: "sagittal"}
    os.makedirs(os.path.join(save_dir, filename), exist_ok=True)
    os.makedirs(os.path.join(save_dir, filename, 't1'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, filename, 't2'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, filename, 't1ce'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, filename, 'flair'), exist_ok=True)
    
    for modalities_ix, modalities in enumerate(npy):
        
        for k in range(128):
            slices = get_slices_from_each_plan(modalities, k)
            if save:
                #   Extract follow 3 dimensions 
                
                for plans_ix, slice in enumerate(slices):
                    if isinstance(slice, np.ndarray):
                        save_name = os.path.join(save_dir, filename, map_modalities_dict[modalities_ix], f"{filename}_{map_plans_dict[plans_ix]}_{k}")
                        np.save(save_name, slice)
                        print(slice.shape, slice.min(), slice.max())
                        # cv2.imwrite(save_name, slice)
                        # print(f"Saved {save_name}")
                    else:
                        print("found None slice")
                    

def normalize(img):
    img = img.astype(np.float32)
    min_value = img.min()
    max_value = img.max()
    img = (img - min_value) / (max_value-min_value)
    img = img * 255.0
    return img


def get_slices_from_each_plan(modalities, k):
    slice_a = modalities[k, :, :]
    slice_c = modalities[:, k, :]
    slice_s = modalities[:, :, k]
    slices = [slice_a, slice_c, slice_s]
    
    # for i in range(len(slices)):
    #     slices[i] = normalize(slices[i])
    #     slices[i] *= -255.0
    #     slices[i] = slices[i].astype(np.uint8)
    #     # print(slices[i].min(), slices[i].max())
    return slices[:1]


def get_array_from_raw(raw_path):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not exists")
    itk_image = sitk.ReadImage(raw_path)
    npy = sitk.GetArrayFromImage(itk_image)
    return npy


def zscore_norm(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        here seg is used to store the zero valued region. The value for that region in the segmentation is -1 by
        default.
        """
        # image = image.astype(self.target_dtype)
        image = image.astype(np.float32)
        seg_tmp = seg
        seg_mask = np.zeros((128, 128, 128))
        for i in seg_tmp[0]:
            seg_mask = seg_mask + i
        seg_mask[seg_mask > 0] = 1

        mask = seg_mask >= 0
        mean = image[mask].mean()
        std = image[mask].std()
        image[mask] = (image[mask] - mean) / (max(std, 1e-8))
        return image


def slice_brats_from_test(save=True):
    npy_dir = Path("/home/dtpthao/workspace/nnUNet/nnunetv2/test_preprocessed/npy")
    seg_dir = Path("/home/dtpthao/workspace/nnUNet/nnunetv2/test_preprocessed/seg")
    npy_files = [x for x in sorted(npy_dir.iterdir())]
    seg_files = [x for x in sorted(seg_dir.iterdir())]
    
    dataset_type="train"
    resize_transform = mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
    for npy_path, seg_path in zip(npy_files, seg_files):
        print(npy_path, seg_path)
        
        data_name = str(npy_path).split('/')[-1][:-4]
        npy = np.load(npy_path)
        # seg = np.load(seg_path)
        slice_brats_utils(
            npy, data_name, resize_transform, 
            save_dir=f"/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brat_slices_new/{dataset_type}",
            save=save
        )
        break


@timing
def slice_brats_from_raw_niigz(save=True):
    dataset_json = "/home/dtpthao/workspace/nnUNet/env/preprocessed/Dataset032_BraTS2018/splits_final.json"
    root_dir = "/home/dtpthao/workspace/nnUNet/env/raw/Dataset032_BraTS2018/imagesTs"
 
    with open(dataset_json, 'r') as f:
        js = json.load(f)
    js = js[0]
    files = Path(root_dir).glob('*_000*.nii.gz')
    # Extract IDs from filenames
    ids = np.unique(["_".join(file.stem.split('_')[:-1]) for file in files]).tolist()
    
    with multiprocessing.get_context("spawn").Pool(default_num_processes) as segmentation_export_pool:
        # for dataset_type, list_file in js.items():
        dataset_type = 'test'
        # list_file = js['val']
        
        for data_name in tqdm(ids):
            npy1 = get_array_from_raw(raw_path=f"{root_dir}/{data_name}_0000.nii.gz")
            npy2 = get_array_from_raw(raw_path=f"{root_dir}/{data_name}_0001.nii.gz")
            npy3 = get_array_from_raw(raw_path=f"{root_dir}/{data_name}_0002.nii.gz")
            npy4 = get_array_from_raw(raw_path=f"{root_dir}/{data_name}_0003.nii.gz")
            npy = np.stack([npy1, npy2, npy3, npy4], axis=0)
            resize_transform = mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
            slice_brats_utils(
                npy, data_name, resize_transform, 
                save_dir=f"/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices_new/{dataset_type}",
                save=save
            )


class BRATSDataset(Dataset):
    def __init__(self, root_dir, train, train_transform=None, val_transform=None, fold=0):
        self.root_dir = root_dir
        self.preprocessed_data_path = "/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset032_BraTS2018/nnUNetPlans_3d_fullres"
        
        self.label_dict = {}
        self.paths = []
        self.labels = []
        
        self.train = train
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        with open("/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset032_BraTS2018/splits_final.json", "r") as f:
            self.dataset_json = json.load(f)[0]
            
        for label in Path(root_dir).iterdir():
            if label.stem not in ['LGG', 'HGG']:
                continue
            for subject in label.iterdir():
                self.label_dict[subject.stem] = 0 if label.stem == 'LGG' else 1

        
        
        if self.train:
            for filename in self.dataset_json["train"]:
                self.paths.append(self.preprocessed_data_path + "/" + filename + ".npy")
                self.labels.append(self.label_dict[filename])
            print("Train LGG: ", len(np.array(self.labels)[np.array(self.labels) == 0]))
        else:
            for filename in self.dataset_json["val"]:
                self.paths.append(self.preprocessed_data_path + "/" + filename + ".npy")
                self.labels.append(self.label_dict[filename])
            print("Val LGG: ", len(np.array(self.labels)[np.array(self.labels) == 0]))
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        
        data = np.load(path)
        if self.train and self.train_transform:
            data = self.train_transform(data)
        if not self.train and self.val_transform:
            data = self.val_transform(data)

        return data.as_tensor(), torch.tensor(float(label))


class BRATS2020Dataset(BRATSDataset):
    def __init__(self, root_dir, train, train_transform=None, val_transform=None, fold=0):
        self.root_dir = root_dir
        self.preprocessed_data_path = "/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset082_BraTS2020/nnUNetPlans_3d_fullres"
        
        self.label_dict = {}
        self.paths = []
        self.labels = []
        
        self.train = train
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        
        import pandas as pd
        df = pd.read_csv("/home/dtpthao/workspace/nnUNet/env/raw/Dataset082_BraTS2020/name_mapping.csv")
        df = df[['BraTS_2020_subject_ID', 'Grade']]
        for _, row in df.iterrows():
            self.label_dict[row['BraTS_2020_subject_ID']] = 0 if row['Grade'] == 'LGG' else 1 
        
        with open("/home/dtpthao/workspace/nnUNet/env/preprocessed/Dataset082_BraTS2020/splits_final.json", "r") as f:
            self.dataset_json = json.load(f)[0]
            
        if self.train:
            for filename in self.dataset_json["train"]:
                self.paths.append(self.preprocessed_data_path + "/" + filename + ".npy")
                self.labels.append(self.label_dict[filename])
            print("Train LGG: ", len(np.array(self.labels)[np.array(self.labels) == 0]))
        else:
            for filename in self.dataset_json["val"]:
                self.paths.append(self.preprocessed_data_path + "/" + filename + ".npy")
                self.labels.append(self.label_dict[filename])
            print("Val LGG: ", len(np.array(self.labels)[np.array(self.labels) == 0]))
        # print(len(self.paths))
        # print(len(self.labels))
        # print(self.paths[0], self.labels[0])


def get_BRATS2020Dataset_dataloader(root_dir, batch_size, num_workers):
    
    train_transform = mt.Compose(
        [
            mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
        ]
    )
    val_transform = mt.Compose(
        [
            mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
        ]
    )
    
    train_dataset = BRATS2020Dataset(root_dir, train=True, train_transform=train_transform, fold=0)
    val_dataset = BRATS2020Dataset(root_dir, train=False, val_transform=val_transform, fold=0)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    sampler = StratifiedBatchSampler(train_dataset.labels, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
    
    
class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        elif isinstance(y, list):
            y = np.array(y)
        assert len(y.shape) == 1, 'label array must be 1D'
        self.n_batches = int(len(y) / batch_size)
        print("Self.n_batches: ", self.n_batches)
        self.skf = StratifiedKFold(n_splits=self.n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        # return len(self.y)
        return self.n_batches


def cv2_loader(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return image


class ImageNetBRATSDataset(Dataset):
    def __init__(self, imagenet_json_path, brats_dir, train=True, train_transform=None, val_transform=None):
        with open(imagenet_json_path, 'r') as f:
            self.imagenet_json = json.load(f)['train' if train else 'val']
        self.brats_dir = Path(brats_dir) / 'train' if train else Path(brats_dir) / 'val'
        self.train = train
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.paths = self.imagenet_json + [x.resolve() for x in self.brats_dir.iterdir()]
        self.labels = np.concatenate([np.zeros(len(self.imagenet_json)), np.ones(len(self.paths) - len(self.imagenet_json))])

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img_path, label = self.paths[idx], self.labels[idx]
        img = cv2_loader(img_path)
        if self.train_transform:
            img = self.train_transform(image=img)["image"]
        if self.val_transform:
            img = self.val_transform(image=img)["image"]
        return img, label


def get_ImageNetBRATSDataset_dataloader(batch_size, num_workers):
    
    # _, model_transform = get_model_and_transform("resnet18")
    import torch
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    # Define the transformation pipeline
    model_transform = A.Compose([
        A.Resize(256, 256, cv2.INTER_LINEAR),
        A.CenterCrop(224, 224, p=1),
        ToTensorV2(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_transform = model_transform
    val_transform = model_transform
    
    brats_folder = "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/brats_slices"
    imagenet_json_path = "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/imagenet_fold_0.json"
    
    train_dataset = ImageNetBRATSDataset(imagenet_json_path=imagenet_json_path, brats_dir=brats_folder, train=True, train_transform=train_transform)
    val_dataset = ImageNetBRATSDataset(imagenet_json_path=imagenet_json_path, brats_dir=brats_folder, train=False, val_transform=val_transform)
    
    sampler = StratifiedBatchSampler(train_dataset.labels, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader
    # return multithreaded_generator, val_loader


if __name__ == '__main__':
    slice_brats_from_raw_niigz(save=True)