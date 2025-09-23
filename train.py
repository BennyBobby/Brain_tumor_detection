from model import BrainTumorClassifier
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np

PROCESSED_DATA_PATH = "data/processed"
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
MODEL_SAVED_PATH = "models/brain_tumor_classifier.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


class TumorDataset(Dataset):
    def __init__(self, img_dir):
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
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = np.transpose(
            image, (2, 0, 1)
        )  # (Height, Width, Channels)=>(Channels, Height, Width)
        image = torch.from_numpy(image).float()
        label = torch.tensor(label).long()
        return image, label
