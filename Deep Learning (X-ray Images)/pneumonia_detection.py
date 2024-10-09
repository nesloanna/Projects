import random
import os
import cv2
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


# ----- Create custom dataset class and a dataloader -----

class CustomDataset(Dataset):

    def __init__(self, input_dir, purpose, device, transform=None):
        self.imgs_path = input_dir
        self.device = device

        # Get list of dataset folders
        dataset = glob.glob(self.imgs_path + purpose)
        dataset_purpose = "".join(dataset).split("/")[1]

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

        print(f"Loaded dataset for {dataset_purpose} (size: {len(self.data)})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)

        # Convert to tensor
        img = torch.tensor(img, dtype=torch.float32)

        # Change shape from (H, W, C) to (C, H, W)
        img = img.permute(2, 0, 1)

        img = self.transform(img)  # Apply transformation to image

        class_id = self.class_map[class_name]
        class_id = torch.tensor(class_id).to(self.device)

        return img, class_id


# ----- Data Augmentation -----

# Transformation for training data
transform_train = v2.Compose(
    [
        v2.Resize((256, 256), antialias=None),  # Resize to 256x256
        v2.Grayscale(1),            # 3 color channels to grayscale (1 channel)
        v2.ToTensor(),                      # Transform images to tensors
        v2.RandomHorizontalFlip(p=0.5),     # Randomly flip images horizontally
        v2.RandomRotation(degrees=(0, 20)),  # Random rotations (0-20 degrees)
        # Normalize values to mean 0.5 and std 0.5
        v2.Normalize((0.5,), (0.5,)),
    ]
)

# Transformation for validation and test data (to fit network input)
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


# ----- Network architecture -----
class xray_model(nn.Module):
    def __init__(self):
        super(xray_model, self).__init__()

        # Convolutional network
        # 16 (batch-size) x 1 x 256 x 256
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=28, kernel_size=3, padding=1)

        # 16 x 28 x 256 x 256  -->  16 x 28 x 128 x 128
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # 16 x 28 x 128 x 128  -->  16 x 64 x 128 x 128
        self.conv2 = nn.Conv2d(
            in_channels=28, out_channels=64, kernel_size=3, padding=1)

        # Fully connected network
        self.fc1 = nn.Linear(64 * 64 * 64, 30)
        self.dropout = nn.Dropout(0.2)  # Dropout
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        # Convolutional network
        x = self.conv1(x)
        x = torch.relu(x)  # ReLU - activation function
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.maxpool1(x)

        # Fully connected network
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = xray_model()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# Path to the saved model
model_path = "xray_model.pth"


# ----- Training the network -----

# Function to train the network

def train_NN(network, num_epochs, learning_rate):

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    current_best_accuracy = 0

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer - ADAM optimizer is faster than SGD
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Chosing the number of epochs for the code to iterate the data

    for epoch in tqdm(range(num_epochs)):
        network.train()  # Train network

        combined_loss = 0
        total_points = 0
        correct = 0

        for images, labels in train_loader:     # Training dataset
            images = images.to(device)          # Sends images to GPU
            labels = labels.to(device)          # Sends labels to GPU

            optimizer.zero_grad()               # Resets gradients
            outputs = network(images)           # Forward pass

            loss = criterion(outputs, labels)   # Compute loss
            combined_loss += loss.item()

            total_points += labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            loss.backward()                     # Backpropagation
            optimizer.step()                    # Update parameters

        train_losses.append(combined_loss / total_points)
        print(
            f"Total points: {total_points}, train_loader: {len(train_loader)}")
        train_accuracy = correct / total_points
        train_accuracies.append(train_accuracy)
        print(
            f"Epoch {epoch+1}/{num_epochs}. Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy:.2%}"
        )

        # --- Validation phase ---

        network.eval()        # Set model to evaluation mode

        correct = 0
        total_points = 0
        combined_loss = 0

        with torch.no_grad():                  # Disable gradient calculation
            for images, labels in val_loader:  # Validation dataset

                images = images.to(device)      # Sends images to GPU
                labels = labels.to(device)      # Sends labels to GPU

                outputs = network(images)       # Forward pass

                loss = criterion(outputs, labels)
                combined_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_points += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = correct / total_points
            val_losses.append(combined_loss / total_points)
            print(f"Validation Accuracy: {val_accuracy:.2%}")

        val_accuracies.append(val_accuracy)

        # Save the model if the current accuracy is better than before
        if val_accuracy > current_best_accuracy:
            torch.save(model.state_dict(), model_path)
            print("Model saved!")
            current_best_accuracy = val_accuracy

    # Plot the loss curve
    x = [i for i in range(num_epochs)]
    print(
        f"x = {x}, {len(x)}, {len(train_losses)}, {len(val_accuracies)}, {len(train_accuracies)}")
    plt.figure(figsize=(9, 6))
    plt.plot(x, train_losses, label="Training Loss")
    plt.plot(x, val_losses, label="Validation Loss")
    plt.plot(x, train_accuracies, label="Training Accuracy")
    plt.plot(x, val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid()
    plt.title("Network Development")
    plt.legend()
    plt.savefig("training_NN.png", dpi=200)
    plt.close()


# Tried different learning rates(lr) - 0.01 and 0.0001
# lr = 0.001 gave the best results
num_epochs = 3

train_NN(model, num_epochs=num_epochs, learning_rate=0.001)

# Loss function
criterion = nn.CrossEntropyLoss()


def plot_confusion_matrix(true, predicted):

    # Plot Confusion matrix to show true and predicted labels
    conf_matrix = confusion_matrix(true, predicted)
    class_names = ["Normal", "Pneumonia"]

    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)

    # Adjust the font size of the axis labels
    plt.xlabel('Predicted label', fontsize=13)
    plt.ylabel('True label', fontsize=13)

    plt.savefig("confusion_matrix.png", dpi=200)
    plt.close()


# ----- Testing the network -----

true_labels = []
predicted_labels = []

model = xray_model()
model.load_state_dict(torch.load(model_path))
model.to(device)


def evaluate_NN(network, test_loader):
    network.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    mislabeled_data = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)      # Sends images to GPU
            labels = labels.to(device)      # Sends labels to GPU

            outputs = model(images)         # Forward pass

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

            # Append misclassified images to list for plotting.
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    image_data = images[i].cpu()
                    true_label = true_labels[i]
                    predicted_label = predicted_labels[i]

                    mislabeled_data.append(
                        {
                            "image": image_data,
                            "true_label": true_label,
                            "predicted_label": predicted_label,
                        }
                    )

    # Test accuracy
    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Plot misclassified images
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    fig.subplots_adjust(hspace=0.5)

    mislabeled_data = mislabeled_data[:10]
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
    plt.savefig("misclassified_images.png", dpi=200)
    plt.close()

    # Plot Confusion matrix to show true and predicted labels
    plot_confusion_matrix(true_labels, predicted_labels)


# Load the best network
model.load_state_dict(torch.load(model_path))

# Show misclassified images
evaluate_NN(model, test_loader)
