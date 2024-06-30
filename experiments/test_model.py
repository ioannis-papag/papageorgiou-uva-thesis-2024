import logging
import numpy as np
#import object_encoder as oe
import torch
#import BlocksGNN as bg
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import BGNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import copy
import random
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import torch.nn as nn
from networks import EmbeddingNet, ReadoutNet
from datasets import FullDataset, SornetDataset
import argparse


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

def compute_per_dimension(predictions, targets):

    num_dimensions = predictions.shape[1]
    
    accuracies = np.zeros(num_dimensions)
    precisions = np.zeros(num_dimensions)
    recalls = np.zeros(num_dimensions)
    
    for i in range(num_dimensions):
        pred_dim = predictions[:, i]
        target_dim = targets[:, i]
        
        accuracies[i] = accuracy_score(target_dim, pred_dim)
        precisions[i] = precision_score(target_dim, pred_dim, zero_division=0)
        recalls[i] = recall_score(target_dim, pred_dim, zero_division=0)
        
    return accuracies, precisions, recalls

def evaluate_model(model, dataloader):
    all_predictions = []
    all_labels = []

    # Loop over batches
    loss_func = nn.BCEWithLogitsLoss()
    losses = []
    counter = 0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            
            # Forward pass to get predictions from the model
            outputs = model(inputs)
            predictions = torch.sigmoid(outputs) >= 0.5
            partial_loss = loss_func(outputs, targets.float())
            losses.append(partial_loss.item())
        
            all_predictions.append(predictions.cpu())
            all_labels.append(targets.cpu())
    per_dim_acc, per_dim_precision, per_dim_recall = compute_per_dimension(torch.cat(all_predictions).numpy(), torch.cat(all_labels).numpy())

    all_predictions = torch.cat(all_predictions).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()


    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = f1_score(all_labels, all_predictions, average='macro')
    print(100*accuracy, 100*precision, 100*recall, 100*f1)
    return 100*accuracy, 100*precision, 100*recall, 100*f1, per_dim_acc, per_dim_precision, per_dim_recall

def evaluate_multiple_models(models, dataloader):
    model_accuracies = []
    model_precisions = []
    model_recalls = []
    model_f1s = []

    model_per_dimension_accuracies = []
    model_per_dimension_precision = []
    model_per_dimension_recall = []
    
    for model in models:

        model.eval()
        
        accuracy, precision, recall, f1, per_dim_accuracy, per_dim_precision, per_dim_recall = evaluate_model(model, dataloader)
        
        model_accuracies.append(accuracy)
        model_precisions.append(precision)
        model_recalls.append(recall)
        model_f1s.append(f1)

        model_per_dimension_accuracies.append(per_dim_accuracy)
        model_per_dimension_precision.append(per_dim_precision)
        model_per_dimension_recall.append(per_dim_recall)

    
    # Calculate mean and standard deviation for metrics across all models
    avg_accuracy = np.mean(model_accuracies)
    std_accuracy = np.std(model_accuracies)
    
    avg_precision = np.mean(model_precisions)
    std_precision = np.std(model_precisions)
    
    avg_recall = np.mean(model_recalls)
    std_recall = np.std(model_recalls)
    
    avg_f1 = np.mean(model_f1s)
    std_f1 = np.std(model_f1s)
    
    avg_per_dimension_accuracy = np.mean(model_per_dimension_accuracies, axis=0)
    std_per_dimension_accuracy = np.std(model_per_dimension_accuracies, axis=0)

    avg_per_dimension_precision= np.mean(model_per_dimension_precision, axis=0)
    std_per_dimension_precision = np.std(model_per_dimension_precision, axis=0)

    avg_per_dimension_recall = np.mean(model_per_dimension_recall, axis=0)
    std_per_dimension_recall = np.std(model_per_dimension_recall, axis=0)
    
    return (avg_accuracy, std_accuracy), (avg_precision, std_precision), (avg_recall, std_recall), (avg_f1, std_f1), (avg_per_dimension_accuracy, std_per_dimension_accuracy), (avg_per_dimension_precision, std_per_dimension_precision), (avg_per_dimension_recall, std_per_dimension_recall)


class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

    
def main():

    model_type = args.model_type
    train_path = args.path
    dataset_type = args.dataset
    model_dir = args.model_dir

    BATCH_SIZE = 32

    if model_type in ['small', 'medium', 'large', 'small_individual', 'medium_individual', 'large_individual']:
        dataset = FullDataset(csv_file = os.path.join(train_path, 'labels.csv'), 
                              img_dir = os.path.join(train_path, 'images'))
    elif model_type == 'sornet':
        dataset = SornetDataset(os.path.join(train_path, 'labels.csv'), 
                                os.path.join(train_path, 'images'), 
                                os.path.join(train_path, 'imgs/imgs.h5'))

    if dataset_type == 'blocks':
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size

        generator = torch.Generator().manual_seed(1)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_loader = DeviceDataLoader(train_loader, device)
        val_loader = DeviceDataLoader(val_loader, device)
        test_loader = DeviceDataLoader(test_loader, device)
    elif dataset_type == 'mnist':
        test_loader = DeviceDataLoader(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False), device)


    obs = next(iter(test_loader))[0]
    input_shape = obs[0].size()
    models = []
    if model_type in ['small', 'medium', 'large']:
        model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
        for i, model_file in enumerate(model_files):
            model_path = os.path.join(model_dir, model_file)
            model_dict = torch.load(model_path)
            model = BGNN.FullPipelineModel(embedding_dim=512, input_dims=input_shape, hidden_dim=512, num_objects=7, size=model_type)
            model.load_state_dict(model_dict)
            models.append(model)

    elif model_type in  ['small_individual', 'medium_individual', 'large_individual']:

        cnn_files = [f for f in os.listdir(os.path.join(model_dir, 'cnns')) if os.path.isfile(os.path.join(model_dir, f))]
        encoder_files = [f for f in os.listdir(os.path.join(model_dir, 'encoders')) if os.path.isfile(os.path.join(model_dir, f))]
        gnn_files = [f for f in os.listdir(os.path.join(model_dir, 'gnns')) if os.path.isfile(os.path.join(model_dir, f))]
        for i, cnn_file in enumerate(cnn_files):
            cnn_path = os.path.join(os.path.join(model_dir, 'cnns'), cnn_file)
            encoder_path = os.path.join(os.path.join(model_dir, 'encoders'), encoder_files[i])
            gnn_path = os.path.join(os.path.join(model_dir, 'gnns'), gnn_files[i])

            model = BGNN.FullPipelineModel(embedding_dim=512, input_dims=input_shape, hidden_dim=512, num_objects=7, size=model_type[:-11], cnn_dir = cnn_path, encoder_dir= encoder_path, gnn_dir= gnn_path)
            models.append(model)
    elif model_type == 'sornet':
        model_files = [f for f in os.listdir(os.path.join(model_dir, 'models')) if os.path.isfile(os.path.join(model_dir, f))]
        rnet_files = [f for f in os.listdir(os.path.join(model_dir, 'rnets')) if os.path.isfile(os.path.join(model_dir, f))]
        for i, model_file in enumerate(model_files):
            model_path = os.path.join(os.path.join(model_dir, 'models'), model_file)
            rnet_path = os.path.join(os.path.join(model_dir, 'rnets'), rnet_files[i])
            model = EmbeddingNet(input_dim=(480,480), patch_size=80, n_objects=7, width=768, layers=12, heads=12)
            r_net = ReadoutNet(d_input=768, d_hidden=512, n_unary=3, n_binary=1)
            model.load_state_dict(torch.load(model_path))
            r_net.load_state_dict(torch.load(rnet_path))

            model_seq = nn.Sequential(model, r_net)
            models.append(model_seq)

    for model in models:
        to_device(model, device)
    (avg_accuracy, std_accuracy), (avg_precision, std_precision), (avg_recall, std_recall), (avg_f1, std_f1), (avg_per_dimension_accuracy, std_per_dimension_accuracy), (avg_per_dimension_precision, std_per_dimension_precision), (avg_per_dimension_recall, std_per_dimension_recall) = evaluate_multiple_models(models, test_loader)

    print(f"Average Accuracy: {avg_accuracy:.3f} ± {std_accuracy:.3f}")
    print(f"Average Precision: {avg_precision:.3f} ± {std_precision:.3f}")
    print(f"Average Recall: {avg_recall:.3f} ± {std_recall:.3f}")
    print(f"Average F1 Score: {avg_f1:.3f} ± {std_f1:.3f}")

    print(f"Average Per-Dimension Accuracy: {avg_per_dimension_accuracy}")
    print(f"Average Per-Dimension Precision: {avg_per_dimension_precision}")
    print(f"Average Per-Dimension Recall: {avg_per_dimension_recall}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-m" , "--model_type", type=str, help="Which model to test. Can be one of: small, medium, large, small_individual, medium_individual, large_individual, sornet", default="small")
    parser.add_argument("-d", "--dataset", type=str, help="Dataset to test. Can be one of: blocks, mnist", required=False, default="blocks")
    parser.add_argument("-md", "--model_dir", type=str, help="Path to models folder. In the case of the individual models, needs to have 3 subfolders named 'cnns', 'encoders', 'gnns'. In the case of sornet needs to have 2 subfolders named 'models', 'rnets'", required=True)
    parser.add_argument("-p", "--path", type=str, help="Path to test dataset", required=True)

    args = parser.parse_args()
    main(args)