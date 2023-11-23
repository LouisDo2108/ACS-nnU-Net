import torch
import torch.nn as nn
import numpy as np
from nnunetv2.tuanluc_dev.dataloaders import get_BRATSDataset_dataloader
from nnunetv2.tuanluc_dev.network_initialization import HGGLGGClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def compute_metrics(labels, predictions):
    """
    Computes various classification metrics given a set of labels and predictions.

    Args:
        labels (np.ndarray): The ground-truth labels.
        predictions (np.ndarray): The predicted labels.

    Returns:
        A dictionary containing the computed metric values.
    """
    metrics = {'accuracy': accuracy_score,
               'precision': precision_score,
               'recall': recall_score,
               'f1': f1_score}

    metric_values = {}
    for metric in metrics:
        metric_fn = metrics[metric]
        metric_values[metric] = metric_fn(labels, predictions)

    return metric_values


def compute_classification_metrics(model, data_loader, device):
    """
    Computes various classification metrics for a given model and data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        device (str): The device to use for evaluation (e.g. 'cpu' or 'cuda').

    Returns:
        A dictionary containing the computed metric values.
    """
    # Set the model to evaluation mode
    model.eval()

    # Initialize the lists to accumulate labels and predictions
    all_labels = []
    all_predictions = []

    # Iterate over the validation data and accumulate labels and predictions
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.float().to(device)
            output = model(data)

            # Convert the output to a binary prediction
            prediction = torch.round(torch.sigmoid(output))

            all_labels.append(target.cpu().numpy())
            all_predictions.append(prediction.cpu().numpy())

    # Concatenate the accumulated labels and predictions
    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    # Compute the ROC AUC score
    roc_auc = roc_auc_score(all_labels, all_predictions)

    # Compute the classification metrics
    metrics = compute_metrics(all_labels, all_predictions)
    metrics['roc_auc'] = roc_auc
    metrics['count_labels'] = np.unique(all_labels, return_counts=True)[1]
    metrics['count_predictions'] = np.unique(all_predictions, return_counts=True)[1]

    return metrics


if __name__ == '__main__':
    pass
    # train_loader, val_loader = get_BRATSDataset_dataloader(
    #     root_dir='/home/dtpthao/workspace/brats_projects/datasets/BraTS_2018/train',
    #     batch_size=4, num_workers=4)
    
    
    # checkpoint_path = "/home/dtpthao/workspace/nnUNet/nnunetv2/tuanluc_dev/results/hgg_lgg/checkpoints/model_5.pt"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model = HGGLGGClassifier(4, 2).to(device)
    # model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    # train_metrics = compute_classification_metrics(model, train_loader, device)
    # val_metrics = compute_classification_metrics(model, val_loader, device)
    
    # print("Train")
    # for k, v in train_metrics.items():
    #     print(f"{k}: {v}")
    
    # print("Val")
    # for k, v in val_metrics.items():
    #     print(f"{k}: {v}")