import random
import os
import pandas as pd
import cv2
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


# # Path to the image folder
# data_path = "data"


# # The input dimension, with a size of 256x256, for future resizing of the images
# input_dim = (256, 256)


# -------- Create custom dataset class and a dataloader --------


class CustomDataset(Dataset):

    def __init__(self, input_dir, purpose, device, transform=None):
        self.imgs_path = input_dir
        self.device = device

        # Get list of dataset folders
        dataset = glob.glob(self.imgs_path + purpose)
        dataset_purpose = "".join(dataset).split("/")[1]

        # dataset_partitions = glob.glob(self.imgs_path + "*")
        print(f"Searching in: {os.path.join(self.imgs_path, purpose, '*')}")

        self.data = []
        for partition_path in dataset:
            # Go into 'normal' and 'pneumonia' class folders
            class_folders = [
                x.replace("\\", "/")
                for x in glob.glob(os.path.join(partition_path, "*/"))
            ]

            for class_folder in class_folders:
                # Get the class name from folder name
                class_name = class_folder.split("/")[-2]

                # Iterate over each image in the class folder (.jpg)
                for img_path in glob.glob(os.path.join(class_folder, "*.jpg")):
                    self.data.append([img_path, class_name])

        self.class_map = {"normal": 0, "pneumonia": 1}
        self.transform = transform

        print(f"Dataset for {dataset_purpose} (size: {len(self.data)})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)

        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32)

        # Change shape from (H, W, C) to (C, H, W)
        img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # print(f"Image shape after resizing: {img.shape}")
        # print(type(img))

        img = self.transform(img)  # Apply transformation

        # Check shape after transformation
        # print(f"Image shape after transform: {img.shape}")
        # print(type(img))

        class_id = self.class_map[class_name]
        # class_id = torch.tensor([class_id])
        class_id = torch.tensor(class_id).to(self.device)

        return img, class_id


# Data Augmentation

transform_train = v2.Compose(
    [
        v2.Resize((256, 256), antialias=None),
        v2.Grayscale(1),
        v2.ToTensor(),
        # v2.RandomErasing(p=0.2, scale=(0.02, 0.1)),
        # v2.GaussianBlur(kernel_size=3),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=(0, 20)),
        # v2.RandomResizedCrop(
        #     size=(256, 256), scale=(0.7, 1), ratio=(1, 1), antialias=None
        # ),
        # v2.ColorJitter(contrast=0.2),
        v2.Normalize((0.5,), (0.5,)),
    ]
)

transform_vt = v2.Compose(
    [
        v2.Resize((256, 256), antialias=None),
        v2.Grayscale(1),
        v2.ToTensor(),
        v2.Normalize((0.5,), (0.5,)),
    ]
)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 16
    input_dir = "data/"
    train_data = CustomDataset(
        input_dir=input_dir,
        purpose="training",
        transform=transform_train,
        device=device,
    )

    val_data = CustomDataset(
        input_dir=input_dir,
        purpose="validation",
        transform=transform_vt,
        device=device,
    )
    test_data = CustomDataset(
        input_dir=input_dir,
        purpose="testing",
        transform=transform_vt,
        device=device,
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


class xray_model(nn.Module):
    def __init__(self):
        super(xray_model, self).__init__()
        # 8 (batch-size) x 3 x 256 x 256
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=28, kernel_size=3, padding=1
        )  # 8 x 28 x 256 x 256
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # 8 x 28 x 128 x 128
        self.conv2 = nn.Conv2d(
            in_channels=28, out_channels=64, kernel_size=3, padding=1
        )  # 8 x 64 x 128 x 128
        self.fc1 = nn.Linear(64 * 64 * 64, 30)
        self.dropout = nn.Dropout(0.2)  # Dropout
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)  # Activation Function
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = xray_model()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
# ADAM optimizer is faster than SGD
# Tried different learning rates(lr) - 0.01 and 0.0001
# lr = 0.001 beacuse it gave the best results

# Path to the saved model
model_path = "xray_model.pth"

# Training the model

train_losses = []
val_losses = []
best_accuracy = 50

# Chosing the number of epochs for the code to iterate the data
num_epochs = 30

for epoch in tqdm(range(num_epochs)):
    model.train()
    combined_loss = 0
    total_points = 0
    correct = 0

    for data, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(data.to(device))

        loss = criterion(outputs, targets)  # calculate loss
        combined_loss += loss.item()
        total_points += targets.size(0)

        _, predicted = torch.max(outputs.data, 1)

        correct += (predicted == targets).sum().item()
        loss.backward()  # backpropagation to calculate the gradients
        optimizer.step()  # update the parameters with respect to gradients

    train_losses.append(combined_loss / total_points)

    train_accuracy = 100 * correct / total_points

    print(
        f"Epoch {epoch+1}/{num_epochs}, Training. Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy:.2f}%"
    )

    # Report the accuracy on validation data

    model.eval()  # set model to evaluation mode
    correct = 0
    total_points = 0
    combined_loss = 0

    with torch.no_grad():  # disable gradient calculation
        for data, targets in val_loader:
            outputs = model(data.to(device))

            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total_points += targets.size(0)
            combined_loss += loss.item()
            correct += (predicted == targets).sum().item()
    val_accuracy = 100 * correct / total_points
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    val_losses.append(combined_loss / total_points)

    # Save the model if the current accuracy is better than before
    if val_accuracy > best_accuracy:
        torch.save(model.state_dict(), model_path)
        print("Model saved!")
        best_accuracy = val_accuracy


# Testing the model

true_labels = []
predicted_labels = []

model = xray_model()
model.load_state_dict(torch.load(model_path))
model.to(device)

# This code snippet does the same as the code above, but with the test data instead
# but the code does so outside of the epoch iteration
model.eval()  # set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        outputs = model(data.to(device))
        loss = criterion(outputs, targets)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        true_labels.extend(targets.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
# sum up the accuracy, to see how accurate the code is functioning
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# --------- Visualization ---------

# Plot Confusion matrix to show true and predicted labels
conf_matrix = confusion_matrix(true_labels, predicted_labels)
class_names = ["Normal", "Pneumonia"]
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.savefig("conf.png", dpi=200)
plt.close()


# Plot the loss curve
plt.figure(figsize=(9, 6))
plt.plot(range(num_epochs), train_losses, label="Loss")
plt.plot(range(num_epochs), train_accuracy, label="Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()
plt.title("Training")
plt.legend()
plt.savefig("training.png", dpi=200)
plt.close()


# Plot the loss curve
plt.figure(figsize=(9, 6))
plt.plot(range(num_epochs), val_losses, label="Loss")
plt.plot(range(num_epochs), val_accuracy, label="Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.grid()
plt.title("Validation")
plt.legend()
plt.savefig("validation.png", dpi=200)
plt.close()


# Function that finds mislabeled data and appends to list
def get_mislabeled(model):
    mislabeled_data = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            incorrect_mask = pred != target

            for i in range(len(data)):
                if incorrect_mask[i]:
                    image_data = data[i].cpu().numpy()
                    true_label = target[i].cpu().item()
                    predicted_label = pred[i].cpu().item()

                    mislabeled_data.append(
                        {
                            "image": image_data,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                        }
                    )

                    if len(mislabeled_data) >= 10:
                        break

            if len(mislabeled_data) >= 10:
                break

    return mislabeled_data


# Function that plots the mislabeled images found in the function before
def plot_mislabeled(mislabeled_data):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.5)

    for i, data_entry in enumerate(mislabeled_data):
        row = i // 5
        col = i % 5
        ax = axes[row, col]

        image = data_entry["image"]
        true_label = data_entry["true_label"]
        predicted_label = data_entry["predicted_label"]
        image = np.transpose(image, (1, 2, 0))
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"True: {true_label}, Predicted: {predicted_label}")
        ax.axis("off")

    plt.tight_layout()


# mislabel_data = get_mislabeled(model)
# plot_mislabeled(mislabel_data)
