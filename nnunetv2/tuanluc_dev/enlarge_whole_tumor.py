import os
from pathlib import Path
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from copy import deepcopy
import numpy as np
from scipy.ndimage import binary_dilation


import numpy as np
from scipy.ndimage import binary_dilation

def enlarge_label_boundary(mask, label, num_pixels):
    
    if len(np.unique(mask)) == 1:
        return mask
    
    # Create a binary mask for the specified label
    label_mask = np.where(mask == label, 1, 0)

    # Create a binary mask for the background
    background_mask = np.where(mask == 0, 1, 0)

    # Perform binary dilation on the label mask
    dilated_mask = binary_dilation(label_mask, iterations=num_pixels)

    # Identify the pixels that are both dilated label and background
    boundary_mask = np.logical_and(dilated_mask, background_mask)

    # Combine the boundary mask with the original mask, preserving other labels
    enlarged_mask = np.where(boundary_mask, label, mask)

    # print("Mask unique values: ", np.unique(mask, return_counts=True))
    # print("Enlarged mask unique values: ", np.unique(enlarged_mask, return_counts=True))
    # print("---")
    
    return enlarged_mask


def enlarge_whole_tumor(filename, input_folder, output_folder, num_pixels):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = deepcopy(b)
    for i in range(len(c)):
        slice = deepcopy(c[i])
        c[i] = enlarge_label_boundary(slice, 2, num_pixels)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))
    
    
if __name__ == "__main__":
    input_path = "/home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/t1_only/fold_0/test_brats_format_copy"
    output_path = "/home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/t1_only/fold_0/test_brats_format_enlarge_wt_1"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    from tqdm import tqdm
    for filename in tqdm(os.listdir(input_path)):
        enlarge_whole_tumor(filename, input_path, output_path, num_pixels=1)

    