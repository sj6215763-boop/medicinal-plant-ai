import os
import json
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


# -----------------------------
# Settings
# -----------------------------

DATASET_PATH = "dataset"
MODEL_PATH = "model/plant_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# -----------------------------
# Image transformations
# -----------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# -----------------------------
# Dataset
# -----------------------------

dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=train_transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# -----------------------------
# Save class names
# -----------------------------

os.makedirs("model", exist_ok=True)

with open(
    "model/class_names.json",
    "w"
) as f:

    json.dump(
        dataset.class_to_idx,
        f
    )


# -----------------------------
# Train/validation split
# -----------------------------

train_size = int(
    0.8 * len(dataset)
)

validation_size = (
    len(dataset) - train_size
)

train_dataset, validation_dataset = torch.utils.data.random_split(
    dataset,
    [train_size, validation_size]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# -----------------------------
# MobileNetV3
# -----------------------------

model = models.mobilenet_v3_small(
    weights="DEFAULT"
)


# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False


# Replace classifier
number_of_classes = len(
    dataset.classes
)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    number_of_classes
)

model = model.to(device)


# -----------------------------
# Loss + optimizer
# -----------------------------

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)


# -----------------------------
# Training
# -----------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(
            outputs.data,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


    train_accuracy = (
        100 * correct / total
    )


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(
                outputs.data,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    validation_accuracy = (
        100 * val_correct / val_total
    )


    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )


# -----------------------------
# Save model
# -----------------------------

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print()
print("================================")
print("Training completed!")
print("Model saved:", MODEL_PATH)
print("Classes:", dataset.classes)
print("================================")