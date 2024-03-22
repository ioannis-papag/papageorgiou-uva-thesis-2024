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
