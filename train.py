import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2
import os
from models import uNet
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from utils import LoadDataset, plot_metrics

def train(model, device, epochs: int=1, learning_rate: float=1e-5, batch_size: int=1):
    
   transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB normalization
        ])
    img_dir = "data/processed_data/img"
    mask_dir = "data/processed_data/mask"

    img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")])
    mask_files = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
    assert len(img_files) == len(mask_files), "Mismatch between images and masks"

    dataset = LoadDataset(img_files, mask_files, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = model
    
    total_params = sum(p.numel() for p in model.parameters()) # Total number of parameters in the model
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"Model Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {total_trainable_params:,}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    for epoch in range(epochs):
        
        model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (Train)", leave=True, ncols=100)
        
        for inputs, labels in train_loop:
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
             
            # Pixel-wise accuracy for binary mask
            preds = (torch.sigmoid(outputs) > 0.5)
            labels_bin = (labels > 0.5)
            correct += (preds == labels_bin).sum().item()
            total += labels.numel()
       
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        train_loop.set_description(f"Epoch {epoch+1}/{epochs} (Train) Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
        
        model.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            
            val_loop = tqdm.tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} (Validation)", leave=False)
            
            for inputs, labels in val_loop:
                
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = (torch.sigmoid(outputs) > 0.5)
                labels_bin = (labels > 0.5)
                val_correct += (preds == labels_bin).sum().item()
                val_total += labels.numel()
                
            val_loss /= len(val_loader.dataset)
            val_acc = val_correct / val_total
            val_loop.set_description(f"Epoch {epoch+1}/{epochs} (Validation) Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        print(f'Epoch {epoch+1}/{epochs}, 'f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, 'f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc            
            torch.save(model.state_dict(), 'best_model.pth')
    
    plot_metrics(train_losses, val_losses, train_accs, val_accs, epochs)
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    return model

if __name__ == '__main__':
    model = uNet(n_channels=3)
    model = model.to(memory_format=torch.channels_last)
    