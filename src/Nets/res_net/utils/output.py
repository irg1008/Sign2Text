from matplotlib import pyplot as plt
import cv2
import numpy as np


def plot_costs(costs):
    """Plot cost curve.

    Args:
        costs (list): A list of costs.
    """
    # Plot cost.
    plt.plot(costs)
    plt.ylabel("Cost")
    plt.xlabel("Epoch")
    plt.show()


def show_image(img_path):
    """Show and image from a path.

    Args:
        img (str): A path to an image.
    """
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()


def imshow(img):
    """Show an image.

    Args:
        img (np.ndarray): An image.
    """
    npimg = img.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
