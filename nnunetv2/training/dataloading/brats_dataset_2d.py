import os
import torch
import numpy as np
from torch.utils.data import Dataset
import json
from pathlib import Path
from glob import glob
import nibabel as nib

class BRATSDataset(Dataset):
    def __init__(self, root_dir,
                       train, 
                       train_transform=None, 
                       val_transform=None, 
                       fold=0, 
                       preprocessed_data_dir="/tmp/htluc/nnunet/nnUNet_preprocessed/Dataset032_BraTS2018"):

        self.root_dir = root_dir
        self.preprocessed_data_path = os.path.join(preprocessed_data_dir,"nnUNetPlans_3d_fullres")
        
        self.label_dict = {}
        mask_dict = {}
        self.paths = []
        self.labels = []
        self.masks_name = []
        
        self.train = train
        self.train_transform = train_transform
        self.val_transform = val_transform
        
        with open(f"{preprocessed_data_dir}/splits_final.json", "r") as f:
            self.dataset_json = json.load(f)[0]
            
        for label in Path(root_dir).iterdir():
            if label.stem not in ['LGG', 'HGG']:
                continue
            for subject in label.iterdir():
                self.label_dict[subject.stem] = 0 if label.stem == 'LGG' else 1
                mask_dict[subject.stem]  = glob(f"{self.root_dir}/**/{subject.stem}/*seg.nii.gz")[0]
                

        
        if self.train:
            for filename in self.dataset_json["train"]:
                self.paths.append(self.preprocessed_data_path + "/" + filename + ".npy")
                self.labels.append(self.label_dict[filename])
                self.masks_name.append(mask_dict[filename])
            print("Train LGG: ", len(np.array(self.labels)[np.array(self.labels) == 0]))
        else:
            for filename in self.dataset_json["val"]:
                self.paths.append(self.preprocessed_data_path + "/" + filename + ".npy")
                self.labels.append(self.label_dict[filename])
                self.masks_name.append(mask_dict[filename])
            print("Val LGG: ", len(np.array(self.labels)[np.array(self.labels) == 0]))
        
    def __len__(self):
        return len(self.paths)
    
    def cal_perc(self, path):
        data = nib.load(path)
        data = data.get_fdata()
        ET = 0
        NCRNET = 0
        for slice in data:
            ET += (slice==4).sum()
            NCRNET += (slice==1).sum()
        TC = ET+NCRNET
        return float(ET/TC), float(NCRNET/TC)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]
        mask_path = self.masks_name[idx]

        percent_ET, percent_NCRNET = self.cal_perc(mask_path)
        data = np.load(path)
        if self.train and self.train_transform:
            data = self.train_transform(data)
        if not self.train and self.val_transform:
            data = self.val_transform(data)
        
        label_rs = {}
        label_rs['isHGG'] = torch.tensor(float(label))
        label_rs['ET'] = torch.tensor(float(percent_ET))
        label_rs['NCR/NET'] = torch.tensor(float(percent_NCRNET))

        return data.as_tensor(), label_rs