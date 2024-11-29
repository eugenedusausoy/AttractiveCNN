import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from torch.cuda.amp import GradScaler, autocast

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load dataset metadata
source_dir = r'C:/Users/edusausoy/Documents/TP3 ComVIS/projet2324/archive/img_align_celeba/img_align_celeba'
attributes_file = r'C:/Users/edusausoy/Documents/TP3 ComVIS/projet2324/archive/list_attr_celeba.csv'
chosen_attribute = "Attractive"

# Preprocess dataset
df = pd.read_csv(attributes_file)
df = df[['image_id', chosen_attribute]].head(20000)
df[chosen_attribute] = (df[chosen_attribute] > 0).astype(int)  # Convert labels to binary

# Define custom dataset
class CelebADataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        label = self.dataframe.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data augmentation and normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(10, shear=5),
    transforms.RandomResizedCrop(224),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Hyperparameters
batch_size = 32
num_epochs = 10
learning_rate = 0.001
num_folds = 5

# Load pre-trained ResNet model (ResNet50)
model = models.resnet50(pretrained=True)

# Fine-tune more layers: Unfreeze all layers (except for the final layer)
for param in model.parameters():
    param.requires_grad = True

# Modify the final fully connected layer for binary classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1)
    # Do not apply sigmoid here, BCEWithLogitsLoss will handle it
)

model = model.to(device)

# Loss and optimizer (Using AdamW for better weight decay handling)
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for numeric stability with mixed precision
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Learning rate scheduler (Cosine Annealing)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# GradScaler for mixed precision training
scaler = GradScaler()

# K-Fold Cross Validation
kfold = KFold(n_splits=num_folds, shuffle=True)

# Arrays to store metrics for learning curves
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# K-Fold Cross Validation Loop
for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
    print(f"Fold {fold+1}/{num_folds}")
    
    train_data = df.iloc[train_idx]
    val_data = df.iloc[val_idx]
    
    train_dataset = CelebADataset(train_data, source_dir, transform)
    val_dataset = CelebADataset(val_data, source_dir, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Early stopping variables
    best_val_loss = float('inf')
    patience = 3
    early_stop_counter = 0

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().view(-1, 1)
            
            optimizer.zero_grad()
            
            # Mixed Precision Training
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid here
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
        
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}")
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        true_labels, predicted_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().view(-1, 1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid here
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
        
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Step the scheduler
        scheduler.step()

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs with no improvement in validation loss.")
            break
    
    # After training, print the classification report and confusion matrix
    print(f"Cohen's Kappa for Fold {fold+1}: {cohen_kappa_score(true_labels, predicted_labels):.4f}")
    print(classification_report(true_labels, predicted_labels))
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for Fold {fold+1}")
    plt.show()

# Plot learning curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title('Loss Curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Val Accuracy")
plt.title('Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
