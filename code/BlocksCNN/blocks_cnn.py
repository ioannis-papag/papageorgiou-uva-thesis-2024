import torch
import torch.nn as nn
import torch.nn.functional as F


class BlocksCNN(nn.Module):
    def __init__(self):
        super(BlocksCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=10, stride=10)

        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=7, kernel_size=1)
    
    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2(x)

        x = torch.sigmoid(x)

        x = F.interpolate(x, size=(480, 480), mode='bilinear', align_corners=False)
        return x



# class BlocksCNN(nn.Module):
#     """CNN encoder, maps observation to obj-specific feature maps."""
    
#     def __init__(self, input_dim, hidden_dim, num_objects):
#         super(BlocksCNN, self).__init__()

#         self.cnn1 = nn.Conv2d(input_dim, hidden_dim, (9, 9), padding=4)
#         self.act1 = nn.LeakyReLU()
#         self.ln1 = nn.BatchNorm2d(hidden_dim)

#         self.cnn2 = nn.Conv2d(hidden_dim, num_objects, (5, 5), stride=5)
#         self.act2 = nn.Sigmoid()

#     def forward(self, obs):
#         h = self.act1(self.ln1(self.cnn1(obs)))
#         print(f"Shape after first convolution: {h.shape}")
#         h = self.act2(self.cnn2(h))
#         print(f"Shape after second conolution: {h.shape}")
#         h = F.interpolate(h, size=(480, 480), mode='bilinear', align_corners=False)
#         return h
    


