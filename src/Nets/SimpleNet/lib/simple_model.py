from typing import Callable
from torch import nn, Tensor
from torch.nn.common_types import _size_3_t


class CNN(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_classes, batch_size, num_frames, image_size, debug=False):
        super(CNN, self).__init__()
        self.batch_size = batch_size
        self.DEBUG = debug

        # General layers.
        self.conv_relu = nn.LeakyReLU()
        self.lin_relu = nn.LeakyReLU()

        hidden_1, hidden_2 = 32, 64

        # Convolutional layers.
        self.conv1 = self._conv_layer_set(num_frames, hidden_1)
        self.conv2 = self._conv_layer_set(hidden_1, hidden_2)

        # Linear layers.
        # -> // 4 for double downsampling
        # -> // 4 for pooling
        linear_1 = (image_size // 4 // 4) ** 2 * hidden_2
        linear_2 = linear_1 // 4
        self.fc1 = self._linear_layer(linear_1, linear_2)
        self.drop = nn.Dropout(p=0.1)
        self.batch_out = nn.BatchNorm1d(linear_2)
        self.fc2 = self._linear_layer(linear_2, image_size)
        self.fc3 = self._linear_layer(image_size, num_classes)
        self.soft = nn.LogSoftmax(dim=1)

    def _conv_layer_set(
        self,
        in_c,
        out_c,
        kernel_size: _size_3_t = (3, 3, 3),
        stride=2,
        padding=1,
        pool: _size_3_t = (1, 2, 2),
    ):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            self.conv_relu,
            nn.MaxPool3d(pool),
            nn.BatchNorm3d(out_c),
        )
        return conv_layer

    def _linear_layer(self, in_c, out_c):
        linear_layer = nn.Sequential(
            nn.Linear(in_c, out_c),
            self.lin_relu,
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
        x = self._assign(x, self.conv2, "conv2")

        x = x.view(x.shape[0], -1)
        if self.DEBUG:
            print("View - ", x.shape)

        x = self._assign(x, self.fc1, "fc1")
        x = self._assign(x, self.batch_out, "batch")
        x = self._assign(x, self.drop, "drop")
        x = self._assign(x, self.fc2, "fc2")
        x = self._assign(x, self.fc3, "fc3")

        x = self._assign(x, self.soft, "soft")

        return x
