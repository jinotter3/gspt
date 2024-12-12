from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from renderer import GaussianRenderer
from dataset import GSCIFAR10



# Example classification network
class GaussianClassifier(nn.Module):
    def __init__(self, num_classes):
        super(GaussianClassifier, self).__init__()
        
        # Embedding MLP for each Gaussian: 8 -> 128
        self.embed = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Classification head: after pooling we have 128 dims
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # x shape: (B, 128, 8)
        B, N, D = x.size()
        
        # Embed each Gaussian
        x = self.embed(x)  # (B, 128, 128)
        
        # Mean pool across the 128 Gaussian dimension
        x = x.mean(dim=1)  # (B, 128)
        
        # Classify
        x = self.classifier(x)  # (B, num_classes)
        
        return x

# Example training code
def train_model(dataset, test_dataset, num_classes=10, num_epochs=20, batch_size=64, lr=1e-3, device='cuda'):
    """
    dataset: An instance of GSCIFAR10 or similar, returning (data, label) pairs where
             data is (128,8) tensor and label is an integer class.
    """
    
    # Prepare dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, loss, optimizer
    model = GaussianClassifier(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for data, labels in dataloader:
            data = data.to(device)   # (B, 128, 8)
            labels = labels.to(device) # (B,)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Tracking performance
            total_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100.0
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")

        if (epoch+1) % 10 == 0:
            evaluate_model(model, test_dataset, batch_size=128, device=device)
    return model


def evaluate_model(model, dataset, batch_size=64, device='cuda'):
    """
    Evaluate the model on a given dataset.

    Args:
        model: Trained PyTorch model.
        dataset: PyTorch Dataset object (e.g., test set).
        batch_size: Batch size for evaluation.
        device: 'cuda' or 'cpu'.

    Returns:
        average_loss (float), accuracy (float, as a percentage)
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, labels in dataloader:
            data = data.to(device)   # (B, 128, 8)
            labels = labels.to(device) # (B,)

            outputs = model(data)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * data.size(0)
            _, preds = torch.max(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += data.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100.0
    print(f"Evaluation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return average_loss, accuracy


if __name__=="__main__":
    train_dataset = GSCIFAR10(dir_path='./cifar10/train', dynamic_loading=False, raw=False, normalization='mean_std', order_random=True, value_jitter=0.01, scaling_jitter=0.01, rotation_jitter=0.01, xy_jitter=0.01)
    test_dataset = GSCIFAR10(dir_path='./cifar10/test', label_path= './cifar10/test', orig_path='./cifar10/test_orig', rendered_path='./cifar10/test_rendered', dynamic_loading=False, raw=False, normalization='mean_std')
    model = train_model(train_dataset, test_dataset=test_dataset, num_classes=10, num_epochs=250, batch_size=128, lr=1e-3, device='cuda')

    avg_loss, acc = evaluate_model(model, test_dataset, batch_size=128, device='cuda')
    
    print("Average Loss: ", avg_loss)
    print("Accuracy: ", acc)
    
    torch.save(model.state_dict(), 'classifier_150.pth')