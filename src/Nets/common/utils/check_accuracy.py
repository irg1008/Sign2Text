from typing import List
from torch.utils.data import DataLoader
from torch.nn import Module
import torch


def check_accuracy(
    loader: DataLoader,
    model: Module,
    classes: List[str],
    device,
    n_batchs=10,
    has_pose=False,
):
    """Check the accuracy of the model on the dataset.

    Args:
        loader (DataLoader): The data loader to use.
        model (Module): The model to use.
        classes (List[str]): The list of classes.
        device (Literal[cuda, cpu]): The device to use.
        n_batchs (int, optional): Number of batches to check. Defaults to 10.
        has_pose (bool, optional): Whether the model has pose information. Defaults to False.
    """
    model.to(device)
    model.eval()

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for i, (videos, loader_targets) in enumerate(loader):
            targets = loader_targets[0] if has_pose else loader_targets
            videos, targets = videos.to(device), targets.to(device)

            scores_1 = model(videos)[0] if has_pose else model(videos)
            _, predictions = scores_1.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

            print(f"Predictions for batch {i+1} ")
            print([classes[int(i)] for i in predictions])

            print(f"Ground truth for batch {i+1}")
            print([classes[int(i)] for i in targets])

            print("---------------------------------\n\n")

            if i > n_batchs:
                break

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
