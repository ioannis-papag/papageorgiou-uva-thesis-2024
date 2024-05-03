import numpy as np
import object_encoder as oe
import torch
#import BlocksGNN as bg
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import BGNN
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import torch.nn as nn


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

model = BGNN.FullPipelineModel()
model.load_state_dict(torch.load(r"C:\Users\johnp\Desktop\thesis\full_model"))

val_dataset = NpyDataset(csv_file=r"C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\test\labels.csv", npy_dir=r"C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\test\masks")
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

print(model)

model.eval()
all_targets = []
all_predictions = []
accuracies = []
with torch.no_grad():
    for samples, labels in val_dataloader:
        val_predictions = torch.sigmoid(model(samples)) >= 0.5
        print(val_predictions)
        labels = labels >= 0.5

        #print(val_predictions)
        
        #print(labels)
        
        #partial_accuracy = accuracy_score(labels.detach().numpy(), val_predictions.detach().numpy())
        #print(partial_accuracy)
        #accuracies.append(partial_accuracy)

        #all_targets.append(labels.cpu())
        #all_predictions.append(val_predictions.cpu())

#all_targets = torch.cat(all_targets).cpu()
#all_predictions = torch.cat(all_predictions).cpu()
#print(accuracies)
#accuracy = np.mean(accuracies)
#print(all_targets.shape)
#print(all_predictions.shape)

#accuracy = accuracy_score(all_targets.detach().numpy(), all_predictions.detach().numpy())
#precision, recall, f1, _ = precision_recall_fscore_support(all_targets.detach().numpy(), all_predictions.detach().numpy(), average='macro')

#print(f"Model Accuracy: {accuracy}")
#print(f"Model Precision: {precision}")
#print(f"Model Recall: {recall}")
#print(f"Model F1 Score: {f1}")