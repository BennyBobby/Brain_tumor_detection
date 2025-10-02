import torch.nn as nn


class BrainTumorClassifier(nn.Module):
    """
    This class defines the architecture of a CNN for brain tumour classification.
    """

    def __init__(self, num_classes: int):
        super(BrainTumorClassifier, self).__init__()
        self.features = nn.Sequential(
            # Conv1: 128x128x3 -> 124x124x64
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Pool1: 124x124x64 -> 41x41x64
            nn.MaxPool2d(kernel_size=3, stride=3),
            # Conv2: 41x41x64 -> 37x37x64
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Pool2: 37x37x64 -> 12x12x64
            nn.MaxPool2d(kernel_size=3, stride=3),
            # Conv3: 12x12x64 -> 9x9x128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Pool3: 9x9x128 -> 4x4x128
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv4: 4x4x128 -> 1x1x128
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),  # To prevent overfitting
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
