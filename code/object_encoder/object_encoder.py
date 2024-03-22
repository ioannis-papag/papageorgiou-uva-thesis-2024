import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ObjectEncoder(nn.Module):
    def __init__(self):
        super(ObjectEncoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(480*480, 512),  # First hidden layer
            nn.ReLU(),
            nn.Linear(512, 512),      # Second hidden layer
            nn.LayerNorm(512),        # LayerNorm before the ReLU
            nn.ReLU()
        )

    def forward(self, x):
        #batch_size, _, num_objects = x.shape
        batch_size, height, width, num_objects = x.size()
        x = x.reshape(batch_size, height * width, num_objects)
        x = x.transpose(1, 2)
        
        outputs = []
        for i in range(num_objects):
            object_output = self.mlp(x[:, i, :])
            outputs.append(object_output.unsqueeze(1))
        
        output = torch.cat(outputs, dim=1)
        
        return output

# filenames = [r"C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\test\masks\mask_1.npy", r"C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\test\masks\mask_2.npy"]  # List of your .npy file paths
# samples = [np.load(fname).reshape(1, 480*480, 7) for fname in filenames]
# batch_data = np.vstack(samples)  # Stack the individual samples into a batch
# batch_tensor = torch.from_numpy(batch_data)  # Convert to a PyTorch tensor

# model = ObjectEncoder()
# output = model(batch_tensor)
# print(output.shape)