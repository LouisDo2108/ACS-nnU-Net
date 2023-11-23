import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from pathlib import Path
from tqdm import tqdm
from functools import partial
from batchgenerators.augmentations.color_augmentations import augment_contrast

import cv2
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torchsummary import summary

from nnunetv2.tuanluc_dev.dataloaders import (
    get_BRATSDataset_dataloader, 
    get_BRATS2020Dataset_dataloader,
    get_ImageNetBRATSDataset_dataloader
)
from nnunetv2.tuanluc_dev.network_initialization import HGGLGGClassifier, ImageNetBratsClassifier
from nnunetv2.tuanluc_dev.utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from thop import profile
from thop import clever_format


class PolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


def plot_loss(train_losses, val_losses, output_folder):
    # Plot the training and validation losses
    epochs = range(5, 5*len(train_losses) + 1, 5)
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    if len(val_losses) == 1:
        plt.legend()
    
    # Save the plot as a PNG picture
    plt.savefig(os.path.join(output_folder, 'loss_plot.png'))


def train(model, train_loader, val_loader, 
          output_folder="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/checkpoints",
          num_epochs=100, learning_rate=0.001):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = PolyLRScheduler(optimizer, learning_rate, num_epochs)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([59.0 / (285.0 - 59.0)]).to(device))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([69.0 / (369.0 - 69.0)]).to(device))
    model.to(device)
    model = torch.compile(model)
     # create output folder if not exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
     # create output folder if not exist
    Path(os.path.join(output_folder, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    train_losses = []
    val_losses = []
    my_augment_contrast = partial(
        augment_contrast, 
        contrast_range=(0.75, 1.25), preserve_range=True, per_channel=True, p_per_channel=0.15
    )
    pbar = tqdm(range(1, num_epochs+1))
    for epoch in pbar:
        
        pbar.set_description(f"Epoch {epoch}")
        model.train()
        train_loss = 0.0
        predictions = []
        true_labels = []
        
        # Train
        for batch_idx, (data, target) in enumerate(train_loader):
            pbar.set_description(f"Epoch {epoch} Batch {batch_idx}")
            temp = []
            for d in data:
                d = my_augment_contrast(d.float())
                d = torch.cat([d] * 3, dim=0)
                cv2.imwrite('/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/test.jpg', d.numpy().transpose(1, 2, 0))
                exit(0)
                temp.append(d)
            data = torch.stack(temp)
            data, target = data.to(device), target.float().to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(1), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pred = torch.sigmoid(output).round().squeeze(1)
            predictions.extend(pred.tolist())
            true_labels.extend(target.tolist())
            
        train_loss /= len(train_loader)
        pbar.set_postfix_str(f"Train loss: {train_loss}")
        log_metrics(output_folder, epoch, train_loss, predictions, true_labels, train=True)
        # validate on validation set every 5 epochs
        if epoch % 1 == 0:
            model.eval()
            val_loss = 0.0
            predictions = []
            true_labels = []
            with torch.no_grad():
                for data, target in val_loader:
                    temp = []
                    for d in data:
                        d = torch.cat([d] * 3, dim=0)
                        temp.append(d)
                    data = torch.stack(temp)
                    data, target = data.float().to(device), target.float().to(device)
                    output = model(data)
                    val_loss += F.binary_cross_entropy_with_logits(output.squeeze(1), target, reduction='sum').item()
                    pred = torch.sigmoid(output).round().squeeze(1)
                    predictions.extend(pred.tolist())
                    true_labels.extend(target.tolist())

            # train_loss /= len(train_loader.dataset)
            val_loss /= len(val_loader.dataset)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            log_metrics(output_folder, epoch, val_loss, predictions, true_labels, train=False)

            # save the model
            torch.save(model.state_dict(), os.path.join(output_folder, 'checkpoints', 'model_{:02d}.pt'.format(epoch)))
            log_loss(output_folder, epoch, val_loss, train=False)
            plot_loss(train_losses, val_losses, output_folder)
            
        scheduler.step(train_loss)
        log_loss(output_folder, epoch, train_loss, train=True)


def log_loss(output_folder, epoch, train_loss, train=False):
    train_val = 'Train' if train else 'Validation'
    with open(os.path.join(output_folder, '{}_loss.txt'.format(train_val)), 'a') as f:
        f.write('Epoch: {} \t{} Loss: {:.6f}\n'.format(epoch, train_val, train_loss))


def log_metrics(output_folder, epoch, loss, predictions, true_labels, train=False):
    train_val = 'Train' if train else 'Validation'
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    roc_auc = roc_auc_score(true_labels, predictions)
            
    with open(os.path.join(output_folder, '{}_metrics.txt'.format(train_val)), 'a') as f:
        f.write('Epoch: {:<5}\n'.format(epoch))
        f.write('{:<10} Average Loss:  {:.4f}\n'.format(train_val, loss))
        f.write('{:<10} Accuracy:      {:.4f}\n'.format(train_val, accuracy))
        f.write('{:<10} Precision:     {:.4f}\n'.format(train_val, precision))
        f.write('{:<10} Recall:        {:.4f}\n'.format(train_val, recall))
        f.write('{:<10} F1 Score:      {:.4f}\n'.format(train_val, f1))
        f.write('{:<10} ROC AUC Score: {:.4f}\n'.format(train_val, roc_auc))


def train_hgg_lgg_classifier(output_folder, custom_network_config_path):
    train_loader, val_loader = get_BRATS2020Dataset_dataloader(
        root_dir='/home/dtpthao/workspace/brats_projects/datasets/BraTS_2018/train',
        batch_size=5, num_workers=16
    )
    
    model = HGGLGGClassifier(5, 2, custom_network_config_path=custom_network_config_path).to(torch.device('cuda'))
    summary(model, (4, 128, 128, 128))
    input = torch.randn([1, 4, 128, 128, 128]).to(torch.device('cuda'))
    macs, params = profile(model, inputs=(input, ))
    macs, params = clever_format([macs*2, params], "%.3f")
    print('[Network %s] Total number of parameters: ' % 'disA', params)
    print('[Network %s] Total number of FLOPs: ' % 'disA', macs)
    exit(0)
    # train(model, train_loader, val_loader, 
    #       output_folder=output_folder, 
    #       num_epochs=100, learning_rate=0.001)
    
    
def train_imagenet_brats_resnet18_encoder():
    train_loader, val_loader = get_ImageNetBRATSDataset_dataloader(
        batch_size=256, num_workers=4
    )
    
    model = ImageNetBratsClassifier(num_classes=2)
    # summary(model, (3, 224, 224))

    train(model, train_loader, val_loader, 
          output_folder="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/resnet18_brats_imagenet_encoder_new", 
          num_epochs=100, learning_rate=0.001)


if __name__ == '__main__':
    set_seed(42)
    train_hgg_lgg_classifier(
        output_folder="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/test",
        custom_network_config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/jcs_acs_resnet18_encoder.yaml"
    )