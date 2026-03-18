
# Vehicle Classification CNN — Google Colab
# Trains a standard CNN (from scratch) on 8 vehicle classes using PyTorch.

import os
import zipfile
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

import torchvision
from torchvision import datasets, transforms

import matplotlib
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATASET_ZIP = "vehicle_classification.zip"
DATASET_DIR = "vehicle_classification"

if os.path.isfile(DATASET_ZIP) and not os.path.isdir(DATASET_DIR):
    print("Unzipping dataset …")
    with zipfile.ZipFile(DATASET_ZIP, "r") as z:
        z.extractall(".")
    print("Done.")

# Check that the dataset folder exists
assert os.path.isdir(DATASET_DIR), (
    f"Dataset directory '{DATASET_DIR}' not found in the current working directory."
)

# Define separate transform pipelines
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# load dataset, transforms applied later
base_dataset = datasets.ImageFolder(root=DATASET_DIR, transform=None)
print(f"Total images : {len(base_dataset)}")
print(f"Classes ({len(base_dataset.classes)}): {base_dataset.classes}")

# 80/20 split
train_size = int(0.8 * len(base_dataset))
val_size = len(base_dataset) - train_size

train_subset, val_subset = random_split(
    base_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]  
        if self.transform is not None:
            image = self.transform(image)
        return image, label


# Apply transforms to splits
train_dataset = TransformSubset(train_subset, train_transform)
val_dataset = TransformSubset(val_subset, val_transform)

# DataLoaders
BATCH_SIZE = 64
NUM_WORKERS = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

class VehicleCNN(nn.Module):

    def __init__(self, num_classes: int = 8):
        super(VehicleCNN, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # Fully-connected head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = (correct / total) * 100.0
    return avg_loss, accuracy

if __name__ == "__main__":

    NUM_EPOCHS = 20

    model = VehicleCNN(num_classes=8).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # track metrics
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print("\n")
    print("Training started")

    for epoch in range(1, NUM_EPOCHS + 1):
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # update lr
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch [{epoch:2d}/{NUM_EPOCHS}] - "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:5.2f}%  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:5.2f}%"
        )

    print("Training complete\n")

    final_train_loss, final_train_acc = evaluate(model, train_loader, criterion, device)
    final_val_loss, final_val_acc = evaluate(model, val_loader, criterion, device)

    print(f"Final Training Accuracy:   {final_train_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")

    epochs_range = range(1, NUM_EPOCHS + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs_range, train_losses, label="Train Loss")
    ax1.plot(epochs_range, val_losses, label="Val Loss")
    ax1.set_title("Loss vs. Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Accuracy
    ax2.plot(epochs_range, train_accuracies, label="Train Acc")
    ax2.plot(epochs_range, val_accuracies, label="Val Acc")
    ax2.set_title("Accuracy vs. Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("\nTraining curves saved to training_curves.png")
    plt.show()
