from typing import List
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np


def normalize(data: List[float]) -> List[float]:
    """Normalize a list of data.

    Args:
        data (List[float]):  The data to normalize.

    Returns:
        List[float]: The normalized data.
    """
    return [x / np.max(data) for x in data]


def plot_train_val_data(
    train: List[float],
    val: List[float],
    ylim=(0, 0),
    ylabel="",
    normalize_data=False,
):
    """

    Args:
        train (List[float]):  The training data.
        val (List[float]): The validation data.
        ylim (tuple, optional): Y axis limits. Defaults to (0, 0).
        ylabel (str, optional): Y axis label. Defaults to "".
        normalize_data (bool, optional): Plot normalized data. Defaults to False.
    """
    x = range(len(train))

    if normalize_data:
        train = normalize(train)
        val = normalize(val)

    plt.plot(x, train, val)

    plt.ylabel(ylabel)
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])

    if ylim != (0, 0):
        plt.ylim(*ylim)

    plt.figure(figsize=(1, 1))
    plt.show()


def plot_tensor(tensor: Tensor, dims=(1, 2, 0)):
    """Plot a tensor.

    Args:
        tensor (Tensor): The tensor to plot.
        dims (tuple, optional): Dimensions to permute plot with numpy. Defaults to (1, 2, 0), Which changes last dimension to first.
    """
    img = tensor.permute(dims).cpu().detach().numpy()
    plt.imshow(img)
    plt.show()
