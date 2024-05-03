import logging
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
import matplotlib.pyplot as plt
from PIL import Image

import torch.nn as nn

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

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_uniform_(m.weight)
#         nn.init.zeros_(m.bias)


def batch_accuracy(predictions, targets):
    # Compare the predictions with the targets element-wise
    correct_predictions = (predictions == targets).sum(dim=1)  # Count the number of correct predictions for each batch
    accuracy = correct_predictions.float() / targets.size(1)  # Calculate accuracy for each batch
    return accuracy

def compute_accuracy(model, dataloader):
    accuracies = []
    #match_count = torch.zeros(1, 70)
    with torch.no_grad():
        for inputs, targets in dataloader:
            
            # Forward pass to get predictions from the model
            predictions = torch.sigmoid(model(inputs)) >= 0.5

            targets = targets >= 0.5
            # Count total correct predictions for the batch
            
            batch_correct = (predictions == targets).float().sum()
            #match_count += (predictions == targets)
            #print(match_count / 1500)
            #print(batch_correct/(70*inputs.size()[0]))
            accuracies.append(batch_correct.cpu() / (70*inputs.size()[0]))



    # Calculate accuracy across all samples
    dataset_accuracy = np.mean(accuracies)

    #return dataset_accuracy, match_count / 1500
    return dataset_accuracy, dataset_accuracy

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

class NpyDataset(Dataset):
    def __init__(self, csv_file, npy_dir):

        self.labels_df = pd.read_csv(csv_file, sep=';')
        self.npy_dir = npy_dir

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):

        npy_path = os.path.join(self.npy_dir, f'mask_{idx+1}.npy')
        #new_order = [1, 2, 3, 4, 5, 6, 0]

        sample = np.load(npy_path).astype(np.float32)
        #sample[..., new_order]
        labels = self.labels_df.loc[idx].values.astype('float32')
        sample = torch.from_numpy(sample)
        labels = torch.from_numpy(labels)
        return sample, labels


class FullDataset(Dataset):
    def __init__(self, csv_file, img_dir):

        self.labels_df = pd.read_csv(csv_file, sep=';')
        self.img_dir = img_dir

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):

        #npy_path = os.path.join(self.npy_dir, f'mask_{idx+1}.npy')
        img_name  = os.path.join(self.img_dir, f'image_{idx+1}.png')
        sample = np.array(Image.open(img_name)).astype(np.float32)
        sample = np.transpose(sample, (2, 0, 1))

        #new_order = [1, 2, 3, 4, 5, 6, 0]

        #sample = np.load(npy_path).astype(np.float32)
        #sample[..., new_order]
        labels = self.labels_df.loc[idx].values.astype('float32')
        sample = torch.from_numpy(sample)
        labels = torch.from_numpy(labels)
        return sample, labels
    
def main():
    BATCH_SIZE = 32

    dataset = FullDataset(csv_file=r"C:\Users\johnp\Desktop\thesis\dataset_blocks_mnist\train\labels.csv", 
                        img_dir=r"C:\Users\johnp\Desktop\thesis\dataset_blocks_mnist\train\images")
    dataloader = DataLoader(dataset, batch_size= BATCH_SIZE, shuffle=True, num_workers=4)
    dataloader = DeviceDataLoader(dataloader, device)

    obs = next(iter(dataloader))[0]
    input_shape = obs[0].size()

    model = BGNN.FullPipelineModel(embedding_dim=512, input_dims=input_shape, hidden_dim=512, num_objects=7)
    #model.apply(weights_init)
    torch.cuda.empty_cache()
    to_device(model, device)
    #model = BGNN.EncoderMLP(input_dim = 480*480, output_dim=512, hidden_dim=512, num_objects=7)

    optimizer = torch.optim.Adam(list(model.parameters()), lr=0.001)


    loss_fn = nn.BCEWithLogitsLoss()
    #model.load_state_dict(torch.load("full_pipeline_kipf_24_04_GPU_small_cnn_3"))
    save_folder = r"C:\Users\johnp\Desktop\thesis\code\logs"

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    log_file = os.path.join(save_folder, 'log_02_05_large_cnn_mnist.txt')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file, 'a'))
    print = logger.info

    val_dataset = FullDataset(csv_file=r"C:\Users\johnp\Desktop\thesis\dataset_blocks_mnist\test\labels.csv", 
                            img_dir=r"C:\Users\johnp\Desktop\thesis\dataset_blocks_mnist\test\images")
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dataloader = DeviceDataLoader(val_dataloader, device)

    print('Starting model training...')
    step = 0


    min_loss = 1e-10
    for epoch in range(100):
        partial_loss = 0
        model.train()
        for i, (samples, labels) in enumerate(dataloader):
            optimizer.zero_grad()

            predictions = model(samples)
            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            partial_loss += loss
            #logger.info(f"====> Epoch: {epoch}  | Batch: {i+1}/{len(dataset) // BATCH_SIZE+1} | Average loss: {loss}")
        logger.info((f"====> Epoch: {epoch} | Average loss: {partial_loss / 157}"))
        dataset_accuracy, distribution = compute_accuracy(model, val_dataloader)
        logger.info(f"Accuracy across the entire dataset:{dataset_accuracy}")
        logger.info("-------------------------------------------------------------------------------------")
        torch.save(model.state_dict(), 'full_pipeline_kipf_02_05_GPU_large_cnn_mnist')
        if partial_loss/157 < min_loss:
            break




    # model.load_state_dict(torch.load("full_pipeline_kipf_15_04")) 
    #df = pd.read_csv(r"C:\Users\johnp\Desktop\thesis\dataset_blocks_unique\train\labels.csv", sep=';')

    # frequencies = df.sum(axis=0) #Calculate frequencies
    # plt.figure(figsize=(20, 15))
    # frequencies.plot(kind='bar', color='skyblue')
    # plt.title('Frequency of a predicate being True')
    # plt.xlabel('Predicate')
    # plt.ylabel('Frequency')
    # plt.xticks(rotation=45)
    # plt.grid(axis='y', linestyle='--')

    # plt.show()


    # actual_labels_df = pd.read_csv(r"C:\Users\johnp\Desktop\thesis\dataset_blocks_unique\labels.csv", sep=';')
    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # Assuming 'x' is already defined as a list of counts from your DataFrame
    # x = list(actual_labels_df.sum(axis=1))
    # x = list(actual_labels_df.sum(axis=1))
    # Set the style of seaborn plot
    #sns.set_context('talk')

    # Create a figure with specified size
    # plt.figure(figsize=(30, 15))

    # # Create the histogram using seaborn
    # ax = sns.histplot(x, color='skyblue', edgecolor='black', binwidth=1)

    # # Set plot title and labels
    # ax.set_title('Histogram of Average True Predicates in a State')
    # ax.set_xlabel('Number of True Predicates')
    # ax.set_ylabel('States')

    # # Annotate the bars with the count of values
    # for p in ax.patches:
    #     ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
    #                 ha='center', va='center', fontsize=20, color='black', xytext=(0, 5),
    #                 textcoords='offset points')

    # # Show the plot
    # plt.show()

    # train_frequencies = actual_labels_df.sum(axis=0) #Calculate frequencies
    # train_frequencies_nonzero = train_frequencies[train_frequencies != 0]/7000

    # plt.figure(figsize=(30, 15))
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.2) 
    # train_frequencies_nonzero.plot(kind='bar', color='skyblue')
    # plt.title('Frequency of a predicate being True (Non-Zero Predicates)', fontsize=18)
    # plt.xlabel('Predicate', fontsize=16)
    # plt.ylabel('Frequency', fontsize=16)
    # plt.xticks(rotation=45, fontsize=12)
    # plt.grid(axis='y', linestyle='--')

    # plt.show()

    #cols = df.columns

    dataset_accuracy, distribution = compute_accuracy(model, val_dataloader)
    print("Accuracy across the entire dataset:", dataset_accuracy)


    # plt.bar(cols, distribution.numpy().flatten())
    # plt.title('Distribution of Matches Across Dimensions')
    # plt.xlabel('Dimension')
    # plt.ylabel('Match Count')
    # plt.show()



    # model.eval()
    # all_targets = []
    # all_predictions = []
    # accuracies = []
    # with torch.no_grad():
    #     for i, (samples, labels) in enumerate(val_dataloader):
    #         val_predictions = torch.sigmoid(model(samples)) >= 0.5
    #         #print(model(samples))
    #         #print(labels)
    #         labels = labels >= 0.5
    #         partial_accuracy = batch_accuracy( val_predictions, labels[:, :21])
    #         print(partial_accuracy)
    #         accuracies.append(partial_accuracy)

    # print(f"Average Accuracy: {np.mean(accuracies)}")


if __name__ == '__main__':
    main()