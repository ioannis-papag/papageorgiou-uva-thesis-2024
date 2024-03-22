import pddlgym
import imageio
from PIL import Image
import os
from pddlgym_planners.ff import FF
from itertools import product
import create_predicates as cp
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import SokobanCNN as scnn

class SokobanDataset(Dataset):
    def __init__(self, image_dir):
        """
        Args:
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    
    def segment_grid(self, image):

        blocks = []

        block_height = image.shape[0] // 9
        block_width  = image.shape[1] // 10

        actual_height = block_height * 9
        actual_width = block_width * 10

        image = image[:actual_height, :actual_width]

        for i in range(9):
            for j in range(10):
                top = i * block_height
                left = j * block_width
                bottom = (i + 1) * block_height if i < 9- 1 else image.shape[0]
                right = (j + 1) * block_width if j < 10 - 1 else image.shape[1]
                
                block = image[top:bottom, left:right]
                blocks.append(block)
        block_tensors = [torch.from_numpy(arr) for arr in blocks]
        return block_tensors
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):

        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = imageio.imread(img_name)
        image_blocks = self.segment_grid(image)

        image_blocks = torch.stack(image_blocks)
        
        return image_blocks

env = pddlgym.make("PDDLEnvSokoban-v0")
obs, debug_info = env.reset()


action = env.action_space.sample(obs)
obs, reward, done, debug_info = env.step(action)

print(obs.literals)


image_dir = 'C:/Users/johnp/Desktop/thesis/code/dataset'
sokoban_dataset = SokobanDataset(image_dir=image_dir)

batch_size = 1 
sokoban_dataloader = DataLoader(sokoban_dataset, batch_size=batch_size, shuffle=False)

model = scnn.SokobanCNN()

for image in sokoban_dataloader:
    
    # 'image' shape: [batch_size, 90, H, W, C]

    first_image_blocks = image[0]  # Shape: [90, H, W, C]
    
    for i, block in enumerate(first_image_blocks):
        print(model(block))
        block = block.numpy()
        plt.imshow(block, interpolation='nearest')
        plt.show()
    break
        #block = block.numpy()
        #plt.imshow(block, interpolation='nearest')
        #plt.show()
        #break
    #break


