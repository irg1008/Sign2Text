from torch import nn
import torch.nn.functional as F
import numpy as np


class SimpleNet(nn.Module):
    def __init__(self, num_classes, batch_size, num_frames, image_size, channels=3):
        super(SimpleNet).__init__()
        self.drop_p = 0.2
        frames_1, frames_2 = 32, 64
        linear_1, linear_2 = 256, 128

        # Common layers.
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=self.drop_p)
        self.pool = nn.MaxPool3d(kernel_size=2)

        # First convolutional layer.
        self.conv1 = self._conv3d(num_frames, frames_1)
        self.batch1 = nn.BatchNorm3d(frames_1)

        # Second convolutional layer.
        self.conv2 = self._conv3d(frames_1, frames_2)
        self.batch2 = nn.BatchNorm3d(frames_2)

        self.view = lambda x: x.view(batch_size, -1)
        # self.lin1 = nn.Linear(frames_1 * , linear_1)
        self.lin2 = nn.Linear(linear_1, linear_2)
        self.lin3 = nn.Linear(linear_2, num_classes)

    def _conv3d(self, in_chn, out_chn, kernel_size=3, stride=2, padding=1):
        return nn.Conv3d(in_chn, out_chn, kernel_size, stride, padding)


class HPNet(nn.Module):
    def __init__(self, num_classes, batch_size, num_frames, image_size, channels=3):
        super(HPNet, self).__init__()

        # => (batch_size, num_frames, channels, image_size, image_size)
        frames_1, frames_2 = 32, 64

        self.conv3d_1 = self._conv_layer_set(
            num_frames, frames_1
        )  # => (batch_size, frames_1, channels - 1, image_size/2, image_size/2)

        self.conv3d_2 = self._conv_layer_set(
            frames_1, frames_2
        )  # => (batch_size, frames_2, channels - 2, image_size/4, image_size/4)

        self.batch = nn.BatchNorm3d(frames_2)

        flatted_value = (channels - 2) * (image_size // 4) ** 2
        self.flat = nn.Flatten(start_dim=2)  # => (batch_size, frames_2, flatted_value)

        self.lin_1 = self._linear_layer_set(
            flatted_value, flatted_value // 2
        )  # => (batch_size, frames_2, flatted_value / 2)

        self.lin_2 = self._linear_layer_set(
            flatted_value // 2, flatted_value // 4
        )  # => (batch_size, frames_2, flatted_value / 4)

        self.lin_3 = nn.Linear(
            flatted_value // 4, num_classes
        )  # => (batch_size, frames_2, num_classes)

        self.soft = nn.LogSoftmax(dim=1)
        self.drop = nn.Dropout(p=0.15)
        self.pool = nn.MaxPool3d(2)

    def _conv_layer_set(
        self, in_channels, out_channels, kernel_size=3, stride=2, padding=1
    ):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
        )

    def _linear_layer_set(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        out = self.conv3d_1(x)
        print("Conv1", out.shape)

        out = self.conv3d_2(out)
        print("Conv2", out.shape)

        # out = self.pool(out)
        # print(out.shape)

        out = self.batch(out)
        print("Batch", out.shape)

        # out = self.flat(out)
        # print(out.shape)

        out = out.view(out.size(0), -1)
        print("View", out.shape)

        out = self.lin_1(out)
        print("Lin 1", out.shape)

        out = self.lin_2(out)
        print("Lin 2", out.shape)

        # out = self.drop(out)

        out = self.lin_3(out)
        print("Lin 3", out.shape)

        out = self.soft(out)
        print("Soft", out.shape)

        return out
