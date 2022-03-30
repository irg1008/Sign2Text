from re import M
from typing import Callable, Tuple, Union
from torch import nn, Tensor
from torch.nn.common_types import _size_3_t
from torchvision.utils import make_grid
from config.torch_config import normalize
import matplotlib.pyplot as plt

# This model is extracted partly from here: https://towardsdatascience.com/pytorch-step-by-step-implementation-3d-convolution-neural-network-8bf38c70e8b3
class CNNNet(nn.Module):
    def __init__(
        self, num_classes, batch_size, num_frames, image_size, channels=3, debug=False
    ):
        super(CNNNet, self).__init__()
        self.batch_size = batch_size
        self.DEBUG = debug

        hidden_1, hidden_2 = 32, 64

        self.conv1 = self._conv_layer_set(num_frames, hidden_1)
        self.conv2 = self._conv_layer_set(hidden_1, hidden_2)

        self.batch1 = nn.BatchNorm3d(hidden_1)
        self.batch2 = nn.BatchNorm3d(hidden_2)

        linear_1 = (image_size // 4) ** 2 * hidden_2
        self.fc1 = self._linear_layer(linear_1, image_size)
        self.fc2 = self._linear_layer(image_size, num_classes)

        self.relu = nn.LeakyReLU()
        self.batch_out = nn.BatchNorm1d(image_size)
        self.drop = nn.Dropout(p=0.15)

        self.soft = nn.LogSoftmax(dim=1)

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=2, padding=1),
            self.relu,
            # nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def _linear_layer(self, in_c, out_c):
        linear_layer = nn.Sequential(
            nn.Linear(in_c, out_c),
            self.relu,
        )
        return linear_layer

    def _assign(self, x: Tensor, fn: Callable[[Tensor], Tensor], name: str):
        out = fn(x)
        if self.DEBUG:
            print(f"{name} - {out.shape}")
        return out

    def forward(self, x):
        if self.DEBUG:
            print("In - ", x.shape)

        x = self._assign(x, self.conv1, "conv1")
        x = self._assign(x, self.batch1, "batch1")
        x = self._assign(x, self.conv2, "conv2")
        x = self._assign(x, self.batch2, "batch2")

        x = x.view(x.shape[0], -1)
        if self.DEBUG:
            print("View - ", x.shape)

        x = self._assign(x, self.fc1, "fc1")
        x = self._assign(x, self.batch_out, "batch")
        x = self._assign(x, self.drop, "drop")
        x = self._assign(x, self.fc2, "fc2")

        x = self._assign(x, self.soft, "soft")

        return x


class HPNet(nn.Module):
    def __init__(
        self, num_classes, batch_size, num_frames, image_size, channels=3, debug=False
    ):
        super(HPNet, self).__init__()
        self.batch_size = batch_size
        self.DEBUG = debug

        # => (batch_size, num_frames, channels, image_size, image_size)
        frames_1, frames_2 = 32, 64

        self.conv3d_1 = self._conv_layer_set(
            num_frames, frames_1, kernel_size=(2, 3, 3)
        )  # => (batch_size, frames_1, channels - 1, image_size/2, image_size/2)

        self.batch_1 = nn.BatchNorm3d(frames_1)

        self.conv3d_2 = self._conv_layer_set(
            frames_1, frames_2
        )  # => (batch_size, frames_2, channels - 2, image_size/4, image_size/4)

        self.batch_2 = nn.BatchNorm3d(frames_2)

        n_convs = 2
        flatted_value = frames_2 * (channels - n_convs) * (image_size // 4) ** 2
        self.flat = nn.Flatten(start_dim=1)  # => (batch_size, frames_2, flatted_value)

        self.lin_1 = self._linear_layer_set(
            flatted_value, flatted_value // 2
        )  # => (batch_size, frames_2, flatted_value / 2)

        self.lin_2 = self._linear_layer_set(
            flatted_value // 2, num_classes
        )  # => (batch_size, frames_2, flatted_value / 4)

        self.soft = nn.LogSoftmax(dim=1)
        # self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout(p=0.15)
        self.pool = nn.MaxPool3d(2)

        # TODO: Estamos teniendo mucho overfittinq ue se puede arreglar usando un dropout, o aumentando la cantiodad de datos que tenemos, o incluso mejorando la calidad de estos.
        # Aquí un link a stackoverflow con tips útiles: https://stackoverflow.com/questions/36139980/prevention-of-overfitting-in-convolutional-layers-of-a-cnn

    def _conv_layer_set(
        self,
        in_channels,
        out_channels,
        kernel_size: _size_3_t = 3,
        stride: _size_3_t = 2,
        padding: _size_3_t = 1,
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

    def _assign(self, x: Tensor, fn: Callable[[Tensor], Tensor], name: str):
        out = fn(x)
        if self.DEBUG:
            print(f"{name} - {out.shape}")
        return out

    def forward(self, x):
        if self.DEBUG:
            print("In - ", x.shape)

        x = self._assign(x, self.conv3d_1, "Conv1")
        x = self._assign(x, self.batch_1, "Batch1")
        x = self._assign(x, self.conv3d_2, "Conv2")
        x = self._assign(x, self.batch_2, "Batch2")
        x = self._assign(x, self.flat, "Flat")

        x = x.view(x.shape[0], -1)

        if self.DEBUG:
            print("View - ", x.shape)

        x = self._assign(x, self.lin_1, "Lin1")
        x = self._assign(x, self.lin_2, "Lin2")
        x = self._assign(x, self.soft, "Soft")

        return x
