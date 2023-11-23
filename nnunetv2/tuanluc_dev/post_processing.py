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
from natsort import natsorted

from tqdm import tqdm
import multiprocessing
import shutil
from multiprocessing.pool import Pool


import numpy as np
from scipy.ndimage import binary_dilation


def post_processing(filename, input_folder, output_folder, num_voxels=150):
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = deepcopy(b)
    count = np.unique(c, return_counts=True)[1]
    if len(count) == 4 and count[-1] < num_voxels:
        print(filename)
        print("Before", count)
        c = np.where(c == 4, 1, c)
        print("After", np.unique(c, return_counts=True))
        print("-------------------------")
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))


def convert_labels_back_to_BraTS_2018_2019_convention(input_folder: str, output_folder: str, num_processes: int = 12):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    result in output_folder
    :param input_folder:
    :param output_folder:
    :return:
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    with multiprocessing.get_context("spawn").Pool(num_processes) as p:
        p.starmap(load_convert_labels_back_to_BraTS, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))


if __name__ == "__main__":
    input_path = "/home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/jcs_baseline/fold_0/test_brats_format_copy"
    output_path = "/home/dtpthao/workspace/nnUNet/env/results/Dataset032_BraTS2018/jcs_baseline/fold_0/test_brats_format_post"
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    for filename in tqdm(natsorted(os.listdir(input_path))):
        if filename.endswith(".nii.gz"):
            post_processing(filename, input_path, output_path, num_voxels=200)
    
    

    