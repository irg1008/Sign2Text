from typing import Tuple
from torch import nn, Tensor
from torch.nn.common_types import _size_3_t
import numpy as np


class CNN(nn.Module):  # pylint disable=too-few-public-methods
    """
    Convolutional neural network.
    """

    def __init__(self, num_classes, num_frames, image_size, num_pose_points):
        super().__init__()

        # Convolutional layer.
        final_channels = 1
        pooling_downsample = 2**2  # 2 time(s) 2 pool downsample
        stride_downsample = 2**2  # 2 time(s) 2 stride downsample

        hidden_1, hidden_2 = 64, 128
        self.convs = nn.Sequential(
            conv_layer_set(num_frames, hidden_1, kernel_size=(2, 3, 3)),
            conv_layer_set(hidden_1, hidden_2),
        )

        # The downsampling scale on convolutional (i.e.: 2 + 2 with stride and pooling)
        downsampling = int(
            np.floor(image_size / (pooling_downsample * stride_downsample))
        )
        linear_1 = (downsampling**2) * final_channels * hidden_2
        linear_2 = linear_1 // 2

        # Dense layer.
        self.drop = nn.Sequential(
            nn.Dropout(p=0.25),
            # nn.BatchNorm1d(linear_2),
        )

        # Fully connected layers (linea + non-linear).
        self.class_fc = nn.Sequential(
            linear_layer(linear_1, linear_2),
            linear_layer(linear_2, num_classes),
            nn.Softmax(dim=1),
        )

        self.pose_fc = nn.Sequential(
            linear_layer(linear_1, num_pose_points),
            nn.Sigmoid();
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Make a forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.drop(x)

        # Class output.
        x1 = self.class_fc(x)

        # Output for poses.
        x2 = self.pose_fc(x)

        return x1, x2


def conv_layer_set(
    in_c: int,
    out_c: int,
    kernel_size: _size_3_t = (3, 3, 3),
    stride=2,
    padding=1,
    pool: _size_3_t = (1, 2, 2),
):
    """Create a convolutional layer set.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        kernel_size (_size_3_t, optional): Convolutional kernel size. Defaults to (3, 3, 3).
        stride (int, optional): Stride for kernel reducing. Defaults to 2.
        padding (int, optional): Padding for kernel pass. Defaults to 1.
        pool (_size_3_t, optional): Pool size for pooling. Defaults to (1, 2, 2).

    Returns:
        Sequential: Convolutional layer set.
    """
    conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.MaxPool3d(pool),
        nn.LeakyReLU(),
        nn.BatchNorm3d(
            out_c, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
    )
    return conv_layer


def linear_layer(in_c: int, out_c: int):
    """Create a linear layer set.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.

    Returns:
        Sequential: Linear layer set.
    """
    linear_layer = nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.ReLU(),
    )
    return linear_layer
