from torch import nn, Tensor
from torch.nn.common_types import _size_3_t


class CNN(nn.Module):
    """
    Convolutional neural network.
    """

    def __init__(self, num_classes, num_frames, image_size):
        super().__init__()

        # Convolutional layer.
        hidden_1, hidden_2 = 32, 64

        self.convs = nn.Sequential(
            conv_layer_set(num_frames, hidden_1),
            conv_layer_set(hidden_1, hidden_2),
        )

        # Dense layer.
        # -> // 4 for double downsampling
        # -> // 4 for pooling
        linear_1 = (image_size // 4 // 4) ** 2 * hidden_2
        linear_2 = linear_1 // 4

        self.dense = nn.Sequential(
            linear_layer(linear_1, linear_2),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(linear_2),
            linear_layer(linear_2, image_size),
            linear_layer(image_size, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Make a forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        x = self.dense(x)
        return x


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
        nn.LeakyReLU(),
        nn.MaxPool3d(pool),
        nn.BatchNorm3d(out_c),
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
        nn.LeakyReLU(),
    )
    return linear_layer
