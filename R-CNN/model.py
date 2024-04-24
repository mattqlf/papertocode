import torch as tc
from torch import nn


class CNN(nn.Module):
    """Implementation of CNN as described in [Krizhevsky et al. 2012] with a few modifications:
    - Replace Local Response Normalization with Batch Normalization
    - Do not use architecture that allows training on multiple GPUs as described in section 3.2
    """

    def __init__(self, feature_dim: int = 4096, classes: int = 0):
        """
        Args:
            feature_dim: the amount of neurons in the FC layers, dim of feature vector
            classes: the amount of classes, if 0, do not do fc3
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.classes = classes

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3), nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=self.feature_dim),  # 20480
            nn.ReLU(inplace=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=self.feature_dim, out_features=self.feature_dim),
            nn.ReLU(),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features=self.feature_dim, out_features=self.classes),
            nn.ReLU(),
            nn.Softmax(),
        )

    def forward(self, x: tc.Tensor) -> tc.Tensor:
        """
        Args:
            x: tensor with shape (N, C, H, W) or (C, H, W)
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # if (C, H, W) -> (1, C, H, W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc1(x)
        x = self.fc2(x)

        if self.classes != 0:
            x = self.fc3(x)

        return x
