import time
import random
import numpy as np
import torch
import os
import yaml
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from copy import deepcopy
from natsort import natsorted
from pathlib import Path
import SimpleITK as sitk

train_no_et = [
    "Brats18_2013_1_1",
    "Brats18_2013_15_1",
    "Brats18_2013_16_1",
    "Brats18_2013_24_1",
    "Brats18_2013_29_1",
    "Brats18_2013_8_1",
    "Brats18_2013_9_1",
    "Brats18_TCIA09_177_1",
    "Brats18_TCIA09_254_1",
    "Brats18_TCIA09_402_1",
    "Brats18_TCIA09_462_1",
    "Brats18_TCIA09_493_1",
    "Brats18_TCIA10_175_1",
    "Brats18_TCIA10_266_1",
    "Brats18_TCIA10_307_1",
    "Brats18_TCIA10_310_1",
    "Brats18_TCIA10_325_1",
    "Brats18_TCIA10_351_1",
    "Brats18_TCIA10_413_1",
    "Brats18_TCIA10_625_1",
    "Brats18_TCIA10_644_1",
    "Brats18_TCIA12_298_1",
    "Brats18_TCIA12_466_1",
    "Brats18_TCIA13_618_1",
    "Brats18_TCIA13_623_1",
    "Brats18_TCIA13_630_1",
    "Brats18_TCIA13_634_1",
    "Brats18_TCIA13_645_1",
]

def post_processing(filename, input_folder, output_folder, num_voxels=200):
    input_path = join(input_folder, filename)
    output_path = join(output_folder, filename)
    
    a = sitk.ReadImage(input_path)
    b = sitk.GetArrayFromImage(a)
    
    c = np.copy(b)
    count = np.bincount(c.flatten())
    
    # print(filename, count)
    # return
    if len(count) == 5 and count[4] < num_voxels:
        c = np.where(c == 4, 1, c)
    
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    
    sitk.WriteImage(d, output_path)
    

def zip_folder(input_folder_path, output_folder_path):
    print("Zip input folder path:", input_folder_path)
    print("Zip output folder path:", output_folder_path)
    shutil.make_archive(output_folder_path, 'zip', input_folder_path)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_param_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def load_yaml_config_file(yaml_file_path):
    """
    Load a YAML file, recursively resolving any included files specified with the "includes" keyword.
    """
    with open(yaml_file_path, "r") as file:
        try:
            yaml_dict = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            return {}

        if "includes" in yaml_dict:
            included_files = yaml_dict.pop("includes")
            if isinstance(included_files, str):
                included_files = [included_files]

            for included_file in included_files:
                if not os.path.isabs(included_file):
                    included_path = os.path.join(os.path.dirname(yaml_file_path), included_file)
                else:
                    included_path = included_file

                if not os.path.exists(included_path):
                    print(f"Included file '{included_path}' does not exist")
                    continue

                included_dict = load_yaml_config_file(included_path)
                yaml_dict = {**included_dict, **yaml_dict}

    return yaml_dict


def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.perf_counter()
        ret = f(*args, **kwargs)
        time2 = time.perf_counter()
        print('%s function took %0.4fs' % (f.__name__, (time2 - time1)))
        return ret
    return wrap


def set_seed(seed):
    # Set the seed for all random number generators
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def print_layers(model, target_layer):
    try:
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                print_layers(module, target_layer)
            
            if isinstance(module, target_layer):
                print(module)
                print('-----------------------')
    except Exception as e:
        print(e)


def create_imagenet_val_json():
    import os
    import json
    from sklearn.model_selection import KFold

    # Path to the ImageNet folder
    imagenet_folder = "/home/dtpthao/workspace/datasets/imagenet_val_2017"

    # Number of folds to use
    num_folds = 5

    # Get list of all image files in the folder
    image_files = [str(x) for x in Path(imagenet_folder).iterdir() if x.is_file() and x.suffix in [".JPEG", ".jpg"]]
    print(len(image_files))
    # # Split the image files into train and validation sets using k-fold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    train_indices, val_indices = list(kf.split(image_files))[0]
    train_files = [image_files[i] for i in train_indices]
    val_files = [image_files[i] for i in val_indices]

    # Create a dictionary containing the paths of train and validation image folders
    data = {
        "train": train_files,
        "val": val_files
    }

    # Save the dictionary as a JSON file
    with open("/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/imagenet_fold_0.json", "w") as f:
        json.dump(data, f)
        

def get_model_and_transform(model_name, pretrained=True):
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm import create_model
    
    model = create_model(model_name, pretrained=pretrained)
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    return model, transform