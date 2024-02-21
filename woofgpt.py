import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import RandomAffine, RandomResizedCrop, RandomHorizontalFlip, ColorJitter
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Enhanced transformations with advanced data augmentation
transform = transforms.Compose([
    RandomResizedCrop(224),
    RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    RandomHorizontalFlip(),
    ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = ImageFolder(root='images', transform=transform)

# Split dataset into training and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet and modify the final layer
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 120)  # Assuming 120 classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

# Function for training and validation
def train_model(model, criterion, optimizer, scheduler, train_loader, val_loader, epochs=50, early_stopping_patience=5):
    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = correct / total
        print(f'Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader)}, Val Accuracy: {val_acc}')

        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_woof_id_model.pth')
            patience_counter = 0  # reset counter if performance improved
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

        # Step the scheduler
        scheduler.step(val_loss/len(val_loader))

train_model(model, criterion, optimizer, scheduler, train_loader, val_loader)

# Model evaluation (optional, extend this as needed)
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds))
    print(confusion_matrix(all_labels, all_preds))

# Load the best model for evaluation
model.load_state_dict(torch.load('best_woof_id_model.pth'))
evaluate_model(model, val_loader)
