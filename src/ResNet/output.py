from matplotlib import pyplot as plt
import cv2
import numpy as np


def plot_costs(costs):
    """_summary_

    Args:
        costs (_type_): _description_
    """
    # Plot cost.
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.show()


def show_image(img):
    """_summary_

    Args:
        img (_type_): _description_
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def imshow(img):
    """_summary_

    Args:
        img (_type_): _description_
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
