import matplotlib.pyplot as plt
from torch import Tensor
import numpy as np


def normalize(data):
    return [x / np.max(data) for x in data]


def plot_costs(train_costs, val_costs, ylim=(0, 10), normalize_data=False):
    x = range(len(train_costs))

    if normalize_data:
        train_costs = normalize(train_costs)
        val_costs = normalize(val_costs)

    plt.plot(x, train_costs, val_costs)

    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation"])

    plt.ylim(*ylim)
    plt.figure(figsize=(1, 1))
    plt.show()


def plot_tensor(tensor: Tensor, dims=(1, 2, 0)):
    img = tensor.permute(dims).cpu().detach().numpy()
    plt.imshow(img)
    plt.show()
