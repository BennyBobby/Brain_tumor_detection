from model import BrainTumorClassifier
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim

PROCESSED_DATA_PATH = "data/processed"
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_SAVED_PATH = "model/brain_tumor_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class TumorDataset(Dataset):
    def __init__(self, img_dir):
        """
        Initialises the dataset by scanning the directory and loading all .npy files.

        Args:
            img_dir (str): Path to the processed data directory.
        """
        self.images = []
        self.labels = []
        tumour_names = sorted(os.listdir(img_dir))
        self.class_map = {name: i for i, name in enumerate(tumour_names)}
        for tumour_name in tumour_names:
            tumour_type_path = os.path.join(img_dir, tumour_name)
            for img_file in os.listdir(tumour_type_path):
                if img_file.endswith(".npy"):
                    image_path = os.path.join(tumour_type_path, img_file)
                    self.images.append(np.load(image_path))
                    self.labels.append(self.class_map[tumour_name])

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        Retrieves the image and its corresponding label at the given index.
        The NumPy array image (H, W, C) is converted to a PyTorch tensor (C, H, W).

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the image tensor (float) and the label tensor (long).
        """
        image = self.images[idx]
        label = self.labels[idx]
        image = np.transpose(
            image, (2, 0, 1)
        )  # (Height, Width, Channels)=>(Channels, Height, Width)
        image = torch.from_numpy(image).float()
        label = torch.tensor(label).long()
        return image, label


def train(model, dataloader, loss_function, optimizer, device):
    """
    Performs one training epoch.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader for the training data.
        loss_function (nn.Module): The loss function.
        optimizer (optim.Optimizer): The optimiser for updating weights.
        device (torch.device): The device (CPU or CUDA) to perform computations on.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, loss_function, device):
    """
    Evaluates the model on a validation dataset.

    Args:
        model (nn.Module): The neural network model to evaluate.
        dataloader (DataLoader): DataLoader for the validation data.
        loss_function (nn.Module): The loss function.
        device (torch.device): The device (CPU or CUDA) to perform computations on.

    Returns:
        tuple: A tuple containing (average_loss, accuracy) on the dataset.
    """
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / len(dataloader.dataset)
    return avg_loss, accuracy


if __name__ == "__main__":
    train_dataset = TumorDataset(os.path.join(PROCESSED_DATA_PATH, "Training"))
    train_ind, val_ind = train_test_split(
        np.arange(len(train_dataset)), test_size=0.2, random_state=0
    )

    train_subset = torch.utils.data.Subset(train_dataset, train_ind)
    val_subset = torch.utils.data.Subset(train_dataset, val_ind)

    train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE)

    num_classes = len(train_dataset.class_map)
    model = BrainTumorClassifier(num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss = train(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_accuracy = validate(model, val_dataloader, loss_fn, device)

        print(
            f"Epoch {epoch+1}/{EPOCHS}: "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val Accuracy: {val_accuracy}"
        )
    if not os.path.exists("model"):
        os.makedirs("model")
    torch.save(model.state_dict(), MODEL_SAVED_PATH)
    print(f"\nModel saved : {MODEL_SAVED_PATH}")
