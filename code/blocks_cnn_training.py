from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import torch.optim as optim
import BGNN as bcnn
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
        mask = np.load(mask_path).astype('float32')
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)

        #mask = mask.permute(1, 2, 0)

        return image, mask
    

model = bcnn.BlocksCNN(input_dim = 4, hidden_dim=30, num_objects=7)
criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.001)



def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available() and False:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

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



def train_model(model, criterion, optimizer, train_loader, num_epochs=1):
    print("Starting training...")
    model.train()  
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(loss.item())
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

train_dataset = ObjectDetectionDataset(r"C:\Users\johnp\Desktop\thesis\dataset_blocks_unique\train\images", r"C:\Users\johnp\Desktop\thesis\dataset_blocks_unique\train\masks", transform=transform, target_transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_loader = DeviceDataLoader(train_loader, device)

test_dataset = ObjectDetectionDataset(r"C:\Users\johnp\Desktop\thesis\dataset_blocks_unique\test\images", r"C:\Users\johnp\Desktop\thesis\dataset_blocks_unique\test\masks", transform=transform, target_transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
test_loader = DeviceDataLoader(test_loader, device)
torch.cuda.empty_cache()
to_device(model, device)
#train_model(model, criterion, optimizer, train_loader, num_epochs=3)

#torch.save(model.state_dict(), 'blocks_cnn_model_short')
model.load_state_dict(torch.load('blocks_cnn_model_short')) 

test_model(model, test_loader)