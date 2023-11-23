import os
import shutil
import json
import glob

# Define the paths to the source and destination directories
src_dir = "/tmp/htluc/nnunet/nnUNet_raw/Dataset032_BraTS2018/imagesTr"
train_dest_dir = "/tmp/htluc/nnunet/fold_0/train"
val_dest_dir = "/tmp/htluc/nnunet/fold_0/val"

# Load the JSON data from a file
with open("/home/dtpthao/workspace/nnUNet/env/preprocessed/Dataset032_BraTS2018/splits_final.json", "r") as f:
    data = json.load(f)

# Loop through each dictionary in the JSON data
for item in data:
    # Loop through each file in the train list and copy it to the train destination directory
    for filename in item["train"]:
        _filename = filename + "*.nii.gz"
        src_path = os.path.join(src_dir, _filename)
        for file in glob.glob(src_path):
            train_dest_path = os.path.join(train_dest_dir, file.split("/")[-1])
            shutil.copy(file, train_dest_path)


    # Loop through each file in the val list and copy it to the val destination directory
    for filename in item["val"]:
        _filename = filename + "*.nii.gz"
        src_path = os.path.join(src_dir, _filename)
        for file in glob.glob(src_path):
            val_dest_path = os.path.join(val_dest_dir, file.split("/")[-1])
            shutil.copy(file, val_dest_path)
    break
