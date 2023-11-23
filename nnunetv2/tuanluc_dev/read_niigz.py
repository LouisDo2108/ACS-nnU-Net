from pathlib import Path
import numpy as np
import nibabel as nib
from skimage import io
from PIL import Image
import monai.transforms as mt
import json
from tqdm import tqdm
import shutil
import cv2


def to_uint8(data):
    """Converts the data to uint8 format."""
    data -= data.min()
    data /= data.max()
    data *= 255
    return data.astype(np.uint8)


def nii_to_jpgs(input_path, output_dir, rgb=False):
    """Converts the given nii.gz file to jpg images.

    Args:
        input_path (str): The path to the nii.gz file.
        output_dir (str): The output directory.
        rgb (bool): Whether to save the image in RGB format.
    """
    id = "_".join(input_path.stem.split('_')[:-1])
    output_dir = Path(output_dir / id)
    data = nib.load(input_path).get_fdata()
    data = np.expand_dims(to_uint8(data), 0)
    resize = mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
    data = resize(data).numpy()[0]
    save_3d(output_dir, rgb, data, True)


def save_3d(output_dir, rgb, data, rot=False):
    for axis, axis_name in enumerate(['sagittal', 'coronal', 'axial']):
        channel_dir = output_dir / f"channel_{axis_name}"
        channel_dir.mkdir(parents=True, exist_ok=True)
        for slice in range(data.shape[axis]):
            slice_data = None
            if axis == 0:
                slice_data = data[slice, :, :]
            elif axis == 1:
                slice_data = data[:, slice, :]
            elif axis == 2:
                slice_data = data[:, :, slice]
            if rgb:
                slice_data = np.stack(3 * [slice_data], axis=2)
            if rot:
                slice_data = np.rot90(slice_data, 3)
                # slice_data = np.fliplr(slice_data)
            output_path = channel_dir / f"channel_{axis_name}_{slice}.jpg"
            slice_data = slice_data.astype(np.uint8)
            io.imsave(output_path, slice_data)


def normalize(img):
    img = img.astype(np.float32)
    min_value = img.min()
    max_value = img.max()
    img = (img - min_value) / (max_value-min_value)
    img = img * 255.0
    return img


def np_to_jpgs(input_path, output_dir, rgb=False, modalties=0):
    id = "_".join(input_path.stem.split('_')[:-1])
    output_dir = Path(output_dir / id)
    data = normalize(np.load(input_path)[modalties])
    resize = mt.Resize((128, 128, 128), size_mode='all', mode="trilinear")
    data = resize(np.expand_dims(data, 0))[0].numpy().transpose(2, 1, 0)
    save_3d(output_dir, rgb, data, rot=True)
    

def get_gen_image(input_path, output_dir, slice):
    id = "_".join(input_path.stem.split('_')[:-1])
    output_dir = Path(output_dir / id)
    # data = Image.open(f"{input_path}/{id}_flair_axial_{slice}.jpg")
    # data.save(f"/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/gen/{id}_flair_axial_{slice}.jpg")
    # return data

            
if __name__ == "__main__":
    
    json_file = Path("/home/dtpthao/workspace/nnUNet/env/preprocessed/Dataset032_BraTS2018/splits_final.json")
    with open(json_file) as f:
        val = json.load(f)[0]['val']
    
    
    # t1 gt(raw nii gz), t1ce gt(raw nii gz), t1ce_preprocesed, t1ce nicegan
    
    ### RAW
    # root_dir = Path('/home/dtpthao/workspace/nnUNet/env/raw/Dataset032_BraTS2018/imagesTr/')
    # for instance in tqdm(val):
    #     print(instance)
    #     for file in root_dir.iterdir():
    #         if instance in file.stem:
    #             input_path = file.resolve()
    #             output_dir = Path('/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/raw_nii_gz/t1')
    #             nii_to_jpgs(input_path, output_dir, rgb=True)
    #     exit(0)
        
    ### PREPROCESSED
    root_dir = Path('/home/dtpthao/workspace/nnUNet/env/preprocessed/Dataset032_BraTS2018/nnUNetPlans_3d_fullres/')
    for instance in tqdm(val):
        for file in root_dir.iterdir():
            if 'seg' not in file.name and file.name.split('.')[-1] == 'npy' and instance in file.stem:
                input_path = file.resolve()
                output_dir = Path('/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/preprocessed/t2')
                np_to_jpgs(input_path, output_dir, rgb=True, modalties=2)
    
    ### GEN
    # root_dir = Path("/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data/brats_t1_to_t1ce_npy/val/image/")
    # output_dir = Path('/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/gen/t1ce')
    # for file in tqdm(val):
    #     for axis_name in ['axial', 'coronal', 'sagittal']:
    #         channel_dir = output_dir / file / f"channel_{axis_name}"
    #         channel_dir.mkdir(parents=True, exist_ok=True)
    #         for i in range(128):
    #             input_path = root_dir / f"{file}_t1ce_{axis_name}_{i}.jpg"
    #             img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)   # reads an image in the BGR format
    #             img = np.fliplr(img)
    #             cv2.imwrite(str(channel_dir / f"channel_{axis_name}_{i}.jpg"), img)
                # shutil.copy(input_path, channel_dir / f"channel_{axis_name}_{i}.jpg")

    # gen_path = Path("/home/dtpthao/workspace/brats_projects/domainadaptation/NICE-GAN-pytorch/brats_generated_data/brats_t2_to_flair_npy/train/image/")
    # get_gen_image(gen_path, "Brats18_TCIA13_653_1", 80)
