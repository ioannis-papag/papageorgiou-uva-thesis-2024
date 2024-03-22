import numpy as np
import object_encoder as oe
import torch
import BlocksGNN as bg
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn

node_loss_fn = nn.BCEWithLogitsLoss()
edge_loss_fn = nn.BCEWithLogitsLoss()

def compute_loss(node_predictions, node_labels, edge_predictions, edge_labels):


    node_loss = node_loss_fn(node_predictions, node_labels)
    edge_loss = edge_loss_fn(edge_predictions, edge_labels)

    total_loss = node_loss + edge_loss
    return total_loss

class NpyDataset(Dataset):
    def __init__(self, csv_file, npy_dir):

        self.labels_df = pd.read_csv(csv_file, sep=';')
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):

        npy_path = os.path.join(self.npy_dir, f'mask_{idx+1}.npy')
        new_order = [1, 2, 3, 4, 5, 6, 0]

        sample = np.load(npy_path)
        sample[..., new_order]
        labels = self.labels_df.loc[idx].values.astype('float32')  # Adjust as per your label columns
        sample = torch.from_numpy(sample)
        labels = torch.from_numpy(labels)
        return sample, labels

dataset = NpyDataset(csv_file="./dataset_blocks_full/train/labels.csv", npy_dir="./dataset_blocks_full/train/masks/")
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)


object_encoder = oe.ObjectEncoder() 
gnn = bg.MessagePassingGNN()
optimizer = torch.optim.Adam(list(object_encoder.parameters()) + list(gnn.parameters()), lr=0.001)

for epoch in range(1):
    for i, (samples, labels) in enumerate(dataloader):
        if i > 10:
            break
        optimizer.zero_grad()
        
        node_labels, edge_labels = labels[:, :21], labels[:, 21:]
        encoded_samples = object_encoder(samples)

        node_predictions, edge_predictions = gnn(encoded_samples)
        
        loss = compute_loss(node_predictions, node_labels, edge_predictions, edge_labels)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item()}")


val_dataset = NpyDataset(csv_file="./dataset_blocks_full/test/labels.csv", npy_dir="./dataset_blocks_full/test/masks/")
#val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=True)
def evaluate_model(dataloader, object_encoder, gnn, threshold=0.5):

    object_encoder.eval() 
    gnn.eval()
    total_accuracy_node = 0
    total_accuracy_edge = 0
    with torch.no_grad():
        for samples, labels in dataloader:
            node_labels, edge_labels = labels[:, :21], labels[:, 21:]
            encoded_samples = object_encoder(samples)
            node_predictions, edge_predictions = gnn(encoded_samples)
            
            
            node_predictions = torch.sigmoid(node_predictions) >= threshold
            edge_predictions = torch.sigmoid(edge_predictions) >= threshold
            
            
            correct_node_predictions = (node_predictions == node_labels).float().mean()
            correct_edge_predictions = (edge_predictions == edge_labels).float().mean()
            
            
            #total_accuracy += (correct_node_predictions + correct_edge_predictions) / 2.0
            total_accuracy_node += correct_node_predictions
            total_accuracy_edge += correct_edge_predictions
            
    #average_accuracy = total_accuracy / len(dataloader)
    return total_accuracy_node, total_accuracy_edge #average_accuracy.item()

# Example usage
validation_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False)
average_accuracy, avg_acc = evaluate_model(validation_dataloader, object_encoder, gnn)
print(f"Validation Accuracy: {average_accuracy, avg_acc:.4f}")

#filenames = [r"C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\test\masks\mask_1.npy", r"C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\test\masks\mask_2.npy"]  # List of your .npy file paths
#samples = []

# for fname in filenames:
#     sample = np.load(fname).reshape(1, 480*480, 7)
#     correct_sample = sample[..., new_order]
#     samples.append(sample)

# #samples = [np.load(fname).reshape(1, 480*480, 7) for fname in filenames]
# batch_data = np.vstack(samples)  # Stack the individual samples into a batch
# batch_tensor = torch.from_numpy(batch_data)  # Convert to a PyTorch tensor

# model = oe.ObjectEncoder()
# output = model(batch_tensor)
# model2 = bg.MessagePassingGNN()
# #node_features = torch.randn(7, 512) # Simulated input
# #output = torch.randn(10, 7, 512)
# #print(batch_node_features.shape)
# #print(output.shape)
# node_outputs, edge_outputs = model2(output)


# print("Node Outputs Shape:", node_outputs.shape) # Should be [21] for 7 nodes each with 3 binary outputs
# print("Edge Outputs Shape:", edge_outputs.shape) # Should be [49] for 49 edges each with 1 binary output
