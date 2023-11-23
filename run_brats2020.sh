#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID # Change according to GPU availability
export CUDA_VISIBLE_DEVICES=0 # Change according to GPU availability
export nnUNet_n_proc_DA=8 # Change according to CPU availability, default is 12

# nnUNet_raw="/tmp/htluc/nnunet/nnUNet_raw/"; export nnUNet_raw
# nnUNet_preprocessed="/tmp/htluc/nnunet/nnUNet_preprocessed"; export nnUNet_preprocessed
# nnUNet_results="/tmp/htluc/nnunet/nnUNet_results"; export nnUNet_results

eval "$(conda shell.bash hook)"
conda activate nnunet
cd /home/dtpthao/workspace/nnUNet/nnunetv2

name="baseline"
trainer="nnUNetTrainer_50epochs_tuanluc"
config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/base.yaml"


# Dataset conversion
python dataset_conversion/Dataset082_BraTS2020.py

# Plan + Preprocess
python experiment_planning/plan_and_preprocess_entrypoints.py \
-d 082 \
-c "3d_fullres" \
-np 12 \
--verify_dataset_integrity \
--clean \
--verbose

# Train
python run/run_training.py 082 $name 0 -num_gpus 1 \
-tr $trainer \
-custom_cfg_path $config_path
# --c # Continue training

# Test
train_test="test"
image_folder=$([ "$train_test" == "train" ] && echo "imagesTr" || echo "imagesTs")
python /home/dtpthao/workspace/nnUNet/nnunetv2/inference/predict_from_raw_data.py \
-i /home/dtpthao/workspace/nnUNet/env/raw/Dataset082_BraTS2020/$image_folder \
-o /home/dtpthao/workspace/nnUNet/env/results/Dataset082_BraTS2020/$name/fold_0/$train_test \
-d 082 \
-tr $trainer \
-c $name \
-f 0 \
-custom_cfg_path $config_path

# Convert back to BraTS2018 format
python dataset_conversion/Dataset082_BraTS2020.py \
--exp-name $name \
--train $train_test \
--post