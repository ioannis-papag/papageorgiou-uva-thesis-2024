from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import torch.optim as optim
import BlocksCNN as bcnn
import torch.nn as nn
import torch


class ObjectDetectionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        self.masks = [f'mask_{i+1}.npy' for i in range(len(self.images))] 
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path)
        mask = np.load(mask_path)
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)

        #mask = mask.permute(1, 2, 0)

        return image, mask
    

model = bcnn.BlocksCNN()
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, criterion, optimizer, train_loader, num_epochs=25):
    model.train()  
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss}')


def test_model(model, test_loader):
    model.eval()  
    total_pixels, correct_pixels = 0, 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = outputs > 0.5  
            correct_pixels += (predicted == labels).sum().item()
            total_pixels += torch.numel(predicted)  

    accuracy = 100 * correct_pixels / total_pixels
    print(f'Accuracy: {accuracy}%')


transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = ObjectDetectionDataset(r'C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\train\images', r'C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\train\masks', transform=transform, target_transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

test_dataset = ObjectDetectionDataset(r'C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\test\images', r'C:\Users\johnp\Desktop\thesis\code\dataset_blocks_full\test\masks', transform=transform, target_transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


train_model(model, criterion, optimizer, train_loader, num_epochs=25)

torch.save(model.state_dict(), 'blocks_cnn_model')

test_model(model, test_loader)