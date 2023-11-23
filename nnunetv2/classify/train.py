import os
import sys

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

from pathlib import Path
root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(root)

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.dataloading.data_loader_2d import get_BRATSDataset_dataloader
from nnunetv2.models.encoder import HGGLGGClassifier
from nnunetv2.training.lr_scheduler import PolyLRScheduler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


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


def train(model, train_loader, val_loader, 
          output_folder="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/checkpoints",
          num_epochs=100, learning_rate=0.001):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = PolyLRScheduler(optimizer, learning_rate, num_epochs)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([59.0 / (285.0 - 59.0)]).to(device))
    model.to(device)
    # model = torch.compile(model)
     # create output folder if not exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
     # create output folder if not exist
    Path(os.path.join(output_folder, 'checkpoints')).mkdir(parents=True, exist_ok=True)
    train_losses = []
    val_losses = []
    
    pbar = tqdm(range(1, num_epochs+1))
    for epoch in pbar:
        
        pbar.set_description(f"Epoch {epoch}")
        model.train()
        train_loss = 0.0
        predictions = []
        true_labels = []
        
        # Train
        for batch_idx, (data, target_dict) in enumerate(train_loader):
            
            pbar.set_description(f"Epoch {epoch} Batch {batch_idx}")
            data = data.to(device)
            target_dict['isHGG'] =  target_dict['isHGG'].float().to(device)
            target_dict['ET'] =  target_dict['ET'].float().to(device)
            target_dict['NCR/NET'] =  target_dict['NCR/NET'].float().to(device)

            optimizer.zero_grad()
            logit_ishgg, logit_et, logit_ncrnet = model(data)

            loss_ishgg = criterion(logit_ishgg.squeeze(1), target_dict['isHGG'])
            loss_et = criterion(logit_et.squeeze(1), target_dict['ET'])
            loss_ncrnet = criterion(logit_ncrnet.squeeze(1), target_dict['NCR/NET'])
            loss = loss_ishgg + loss_et + loss_ncrnet

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            pred = torch.sigmoid(logit_ishgg).round().squeeze(1)
            predictions.extend(pred.tolist())
            true_labels.extend(target_dict['isHGG'].tolist())
            
        train_loss /= len(train_loader)
        pbar.set_postfix_str(f"Train loss: {train_loss}")
        log_metrics(output_folder, epoch, train_loss, predictions, true_labels, train=True)
        # validate on validation set every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_loss = 0.0
            predictions = []
            true_labels = []
            with torch.no_grad():
                for data, target_dict in val_loader:
                    data = data.to(device)
                    target_dict['isHGG'] =  target_dict['isHGG'].float().to(device)
                    target_dict['ET'] =  target_dict['ET'].float().to(device)
                    target_dict['NCR/NET'] =  target_dict['NCR/NET'].float().to(device)
                    logit_ishgg, logit_et, logit_ncrnet = model(data)
                    val_loss += F.binary_cross_entropy_with_logits(logit_ishgg.squeeze(1), target_dict['isHGG'], reduction='sum').item()
                    pred = torch.sigmoid(logit_ishgg).round().squeeze(1)
                    predictions.extend(pred.tolist())
                    true_labels.extend(target_dict['isHGG'].tolist())

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


def train_hgg_lgg_classifier(output_folder, custom_network_config_path):
    train_loader, val_loader = get_BRATSDataset_dataloader(
        root_dir='/home/dtpthao/workspace/brats_projects/datasets/BraTS_2018/train',
        batch_size=4, num_workers=8)
    
    model = HGGLGGClassifier(4, 2, custom_network_config_path=custom_network_config_path).to(torch.device('cuda'))
    train(model, train_loader, val_loader, 
          output_folder=output_folder, 
          num_epochs=100, learning_rate=0.001)

if __name__=='__main__':
    train_hgg_lgg_classifier(
        output_folder="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/cls_hgglgg_etncrnet",
        custom_network_config_path="/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/configs/hgg_lgg_acs_resnet18_encoder_all.yaml"
    )