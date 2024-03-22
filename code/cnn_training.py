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
import torch
import torch.nn as nn
import torch.optim as optim


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def main():
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    train_dataset = datasets.ImageFolder(root="C:/Users/johnp/Desktop/thesis/code/cnn_dataset/train", transform=transform)


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    test_dataset = datasets.ImageFolder(root="C:/Users/johnp/Desktop/thesis/code/cnn_dataset/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)


    model = scnn.SokobanCNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total} %')

    torch.save(model.state_dict(), 'sokoban_cnn_model')

def load_model():
    transform = transforms.Compose([
        transforms.ToTensor(), 
    ])
    test_dataset = datasets.ImageFolder(root="C:/Users/johnp/Desktop/thesis/code/cnn_dataset/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    model = scnn.SokobanCNN()
    model.load_state_dict(torch.load('sokoban_cnn_model'))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total} %')

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    #main()
    load_model()
