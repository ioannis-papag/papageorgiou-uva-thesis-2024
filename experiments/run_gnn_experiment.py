import logging
import numpy as np
import torch
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
import argparse
from datasets import GNNDataset

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



def evaluate_model(model_encoder, model_gnn, dataloader):
    model_encoder.eval()
    model_gnn.eval()

    loss = []
    accuracies = []

    loss_func = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):

            predictions = model_gnn(model_encoder(inputs))
            partial_loss = loss_func(predictions.float(), targets.float())
            predictions = torch.sigmoid(predictions) >= 0.5
            targets = targets >= 0.5
            
            batch_correct = (predictions == targets).float().sum()



            loss.append(partial_loss.cpu())

            accuracies.append(batch_correct.cpu() / (70*inputs.size()[0]))

    dataset_accuracy = np.mean(accuracies)
    validation_loss = np.mean(loss)

    return validation_loss, dataset_accuracy

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
    model_type = args.model
    train_path = args.path
    dataset_type = args.dataset

    BATCH_SIZE = 32
    dataset = GNNDataset(csv_file = os.path.join(train_path, 'labels.csv'), 
                    mask_h5_file = os.path.join(train_path, 'imgs/imgs.h5'))
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


    log_file = f'logs/gnn_{model_type}_{dataset_type}.log'
    logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    handlers=[
                        logging.FileHandler(log_file),
                        logging.StreamHandler()
                    ])


    for experiment in range(num_runs):

        set_seed(experiment+1)

        if model_type == 'small':
            model_encoder = BGNN.EncoderMLP(input_dim=48*48, output_dim=512, hidden_dim=512, num_objects=7)
        elif model_type == 'medium':
            model_encoder = BGNN.EncoderMLP(input_dim=96*96, output_dim=512, hidden_dim=512, num_objects=7)
        elif model_type == 'large':
            model_encoder = BGNN.EncoderMLP(input_dim=480*480, output_dim=512, hidden_dim=512, num_objects=7)
        else:
            print("Invalid model type.")
            exit()
        model_gnn = BGNN.BlocksGNN(input_dim=512, hidden_dim=512, num_objects=7)
        torch.cuda.empty_cache()
        to_device(model_encoder, device)
        to_device(model_gnn, device)
        optimizer = torch.optim.Adam(list(model_encoder.parameters()) + list(model_gnn.parameters()), lr=0.001)


        loss_fn = nn.BCEWithLogitsLoss()

        best_loss = float('inf')
        best_model_weights = None
        patience = 15

        print(f'Starting model {experiment+1} training...')
        for epoch in range(200):
            partial_loss = 0
            model_encoder.train()
            model_gnn.train()
            for i, (samples, labels) in enumerate(train_loader):

                optimizer.zero_grad()

                embeddings = model_encoder(samples)
                predictions = model_gnn(embeddings)
                loss = loss_fn(predictions, labels)
                loss.backward()
                optimizer.step()
                partial_loss += loss

            logging.info((f"====> Epoch: {epoch+1} | Average loss: {partial_loss / len(train_loader)}"))
            val_loss, dataset_accuracy = evaluate_model(model_encoder, model_gnn, test_loader) 
            logging.info(f"Accuracy across the entire dataset:{dataset_accuracy}")
            logging.info(f"Validation Loss:{val_loss}")
            logging.info("-------------------------------------------------------------------------------------")
            torch.save(model_encoder.state_dict(), f'gnn_{model_type}_{dataset_type}_encoder_run_{experiment+1}')
            torch.save(model_gnn.state_dict(), f'gnn_{model_type}_{dataset_type}_gnn_run_{experiment+1}')
            if dataset_accuracy == 1:
                best_model_encoder_weights = copy.deepcopy(model_encoder.state_dict())
                best_model_gnn_weights = copy.deepcopy(model_gnn.state_dict())  
                break
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_encoder_weights = copy.deepcopy(model_encoder.state_dict())
                best_model_gnn_weights = copy.deepcopy(model_gnn.state_dict())      
                patience = 15
            else:
                patience -= 1
                if patience == 0:
                    break
        model_encoder.load_state_dict(best_model_encoder_weights)
        model_gnn.load_state_dict(best_model_gnn_weights)
        torch.save(model_encoder.state_dict(), f'gnn_{model_type}_{dataset_type}_encoder_run_{experiment+1}')
        torch.save(model_gnn.state_dict(), f'gnn_{model_type}_{dataset_type}_gnn_run_{experiment+1}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument("-n" , "--runs", type=int, help="Indicates how many runs will be executed.", default=1)
    parser.add_argument("-d" , "--dataset", type=str, help="Which dataset to execute. Can be one of: blocks, mnist", default="blocks")
    parser.add_argument("-m", "--model", type=str, help="Type of model. Can be one of: small, medium, large", required=False, default="small")
    parser.add_argument("-p", "--path", type=str, help="Path to training folder", required=True)

    args = parser.parse_args()
    main(args)