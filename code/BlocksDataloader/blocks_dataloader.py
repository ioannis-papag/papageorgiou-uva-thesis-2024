import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class NPYDataset(Dataset):
    def __init__(self, npy_file_paths, labels_csv_path):

        self.npy_file_paths = npy_file_paths
        self.labels = pd.read_csv(labels_csv_path).values

    def __len__(self):

        return len(self.npy_file_paths)

    def __getitem__(self, idx):
        """
        Retrieves the idx-th sample from the dataset.
        
        :param idx: Index of the sample to retrieve.
        :return: A tuple (sample, label) where sample is the loaded .npy data, and label is the corresponding label tensor.
        """
        # Load the sample
        sample_path = self.npy_file_paths[idx]
        sample = np.load(sample_path)  # Sample shape: (480, 480, 8)
        sample = sample.astype(np.float32).reshape(1, -1, 7)  # Reshape to (1, 480*480, 8) and ensure float32 dtype
        
        # Convert to torch tensor
        sample_tensor = torch.from_numpy(sample)

        # Retrieve and convert the label for this sample
        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return sample_tensor, label_tensor

# Define your paths and create the DataLoader
npy_file_paths = ['path_to_sample1.npy', 'path_to_sample2.npy', ...]  # List your .npy file paths here
labels_csv_path = 'path_to_labels.csv'
dataset = NPYDataset(npy_file_paths, labels_csv_path)

# Adjust batch_size according to your training setup and available memory
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)