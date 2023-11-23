#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
export CUDA_VISIBLE_DEVICES=1 # Change according to GPU availability
export nnUNet_n_proc_DA=8 # Change according to CPU availability, default is 12

eval "$(conda shell.bash hook)"
conda activate nnunet
cd /home/dtpthao/workspace/nnUNet/

name="acs_resnet18"
trainer="nnUNetTrainer_50epochs_tuanluc"
config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/acs_resnet18.yaml"
fold=0

# Train
python nnunetv2/run/run_training.py 032 $name 0 -num_gpus 1 \
-tr $trainer \
-custom_cfg_path $config_path
# --c # Continue training

# Test
train_test="test"
image_folder=$([ "$train_test" == "train" ] && echo "imagesTr" || echo "imagesTs")
python nnunetv2/inference/predict_from_raw_data.py \
-i env/raw/Dataset032_BraTS2018/$image_folder \
-o env/results/Dataset032_BraTS2018/$name/fold_$fold/$train_test \
-d 032 \
-tr $trainer \
-c $name \
-f $fold \
-custom_cfg_path $config_path

# Convert back to BraTS2018 format
python nnunetv2/dataset_conversion/Dataset032_BraTS2018.py \
--exp-name $name \
--train $train_test \
--post \
--fold $fold