import logging
import numpy as np
import torch
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from networks import EmbeddingNet, ReadoutNet
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import copy
import random
import argparse
from datasets import SornetDataset

import torch.nn as nn

import random


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)



def evaluate_model(model, r_net, dataloader):
    all_predictions = []
    all_labels = []

    # Loop over batches
    loss_func = nn.BCEWithLogitsLoss()
    losses = []
    counter = 0
    with torch.no_grad():
        for images, patches, targets in dataloader:
            outputs, _ = model(images, patches)
            outputs = r_net(outputs)

            partial_loss = loss_func(outputs, targets.float())
            losses.append(partial_loss.cpu().item())
            predictions = outputs > 0
            all_predictions.append(predictions.cpu())
            all_labels.append(targets.cpu())

    all_predictions = torch.cat(all_predictions).numpy().flatten()
    all_labels = torch.cat(all_labels).numpy().flatten()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)

    return 100*accuracy, np.mean(losses)

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

    num_runs = args.runs
    train_path = args.path
    dataset_type = args.dataset

    BATCH_SIZE = 32
    dataset = SornetDataset(os.path.join(train_path, 'labels.csv'), os.path.join(train_path, 'images'), os.path.join(train_path, 'imgs/imgs.h5'))
    
    if dataset_type == "blocks":

        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(1)
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)
        device = get_default_device()
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_loader = DeviceDataLoader(train_loader, device)
        test_loader = DeviceDataLoader(test_loader, device)

    elif dataset_type == "mnist":


        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        generator1 = torch.Generator().manual_seed(1)
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator1)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        train_loader = DeviceDataLoader(train_loader, device)
        test_loader = DeviceDataLoader(test_loader, device)

    else:
        print("Invalid dataset type.")
        exit()



    obs = next(iter(train_loader))[0]
    input_shape = obs[0].size()

    save_folder = f"logs"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    log_file = f'logs/sornet_{dataset_type}.log'
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])


    for experiment in range(num_runs):

        set_seed(experiment+1)



        torch.cuda.empty_cache()
        model = EmbeddingNet(input_dim=(480,480), patch_size=80, n_objects=7, width=768, layers=12, heads=12)

        r_net = ReadoutNet(d_input=768, d_hidden=512, n_unary=3, n_binary=1)
        to_device(model, device)
        to_device(r_net, device)
        best_loss = float('inf')
        best_model_weights = None
        best_rnet_weights = None
        patience = 15

        optimizer = torch.optim.Adam(list(model.parameters()) + list(r_net.parameters()), lr=0.0001)

        loss_fn = nn.BCEWithLogitsLoss()

        print(f'Starting model {experiment+1} training...')
        for epoch in range(200):
            partial_loss = 0
            model.train()
            for i, (images, patches, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                outputs, _ = model(images, patches)
                outputs = r_net(outputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                partial_loss += loss

            logging.info((f"====> Epoch: {epoch} | Average loss: {partial_loss / len(train_loader)} "))
            val_accuracy, val_loss = evaluate_model(model, r_net, test_loader)
            logging.info(f"Accuracy across the test dataset:{val_accuracy} | Average validation Loss: {val_loss}")
            torch.save(model.state_dict(), f'sornet_{dataset_type}_model_run_{experiment+1}')
            torch.save(r_net.state_dict(), f'sornet_{dataset_type}_rnet_run_{experiment+1}')
            if val_accuracy == 100:
                best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here
                best_rnet_weights = copy.deepcopy(r_net.state_dict())  # Deep copy here  
                break
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here
                best_rnet_weights = copy.deepcopy(r_net.state_dict())  # Deep copy here      
                patience = 15  # Reset patience counter
            else:
                patience -= 1
                if patience == 0:
                    break
        model.load_state_dict(best_model_weights)
        r_net.load_state_dict(best_rnet_weights)
        torch.save(model.state_dict(), f'sornet_{dataset_type}_model_run_{experiment+1}')
        torch.save(r_net.state_dict(), f'sornet_{dataset_type}_rnet_run_{experiment+1}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("-n" , "--runs", type=int, help="Indicates how many runs will be executed.", default=1)
    parser.add_argument("-d" , "--dataset", type=str, help="Which dataset to execute. Can be one of: blocks, mnist", default="blocks")
    parser.add_argument("-p", "--path", type=str, help="Path to training folder", required=True)

    args = parser.parse_args()
    main(args)