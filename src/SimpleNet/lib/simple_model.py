from torch import nn
import torch.nn.functional as F
import numpy as np


class HPNet(nn.Module):
    def __init__(self, num_classes, batch_size, num_frames, image_size, channels=3):
        super(HPNet, self).__init__()

        # => (batch_size, num_frames, channels, image_size, image_size)

        self.conv3d_1 = self._conv_layer_set(
            num_frames, 32
        )  # => (batch_size, 32, channels - 1, image_size/2, image_size/2)

        self.conv3d_2 = self._conv_layer_set(
            32, 64
        )  # => (batch_size, 64, channels - 2, image_size/4, image_size/4)

        self.batch = nn.BatchNorm3d(batch_size)

        flatted_value = (channels - 2) * (image_size // 4) ** 2
        self.flat = nn.Flatten(start_dim=2)  # => (batch_size, 64, flatted_value)

        self.lin_1 = self._linear_layer_set(
            flatted_value, flatted_value // 2
        )  # => (batch_size, 64, flatted_value / 2)

        self.lin_2 = self._linear_layer_set(
            flatted_value // 2, flatted_value // 4
        )  # => (batch_size, 64, flatted_value / 4)

        self.lin_3 = nn.Linear(
            flatted_value // 4, num_classes
        )  # => (batch_size, 64, num_classes)

        self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.15)

    def _conv_layer_set(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=1
    ):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            # nn.MaxPool3d(2)
        )

    def _linear_layer_set(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # print(x.shape)
        out = self.conv3d_1(x)
        out = self.drop(out)
        # print(out.shape)
        out = self.conv3d_2(out)
        out = self.drop(out)
        # print(out.shape)
        out = self.flat(out)
        # print(out.shape)
        out = self.lin_1(out)
        # print(out.shape)
        out = self.lin_2(out)
        # print(out.shape)
        out = self.lin_3(out)
        # print(out.shape)
        out = self.soft(out)
        # out = self.batch(out)
        # print(out.shape)
        return out.view(out.size(0), -1)
