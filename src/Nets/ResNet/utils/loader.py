from typing import List, Tuple
import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_dataset(
    data_dir: str, transform: Compose
) -> Tuple[datasets.ImageFolder, List[str]]:
    """Loads the dataset and returns it with all the classes.

    Args:
        data_dir (str): The path to the dataset.
        transform (Compose): The transform to apply to the dataset.

    Returns:
        Tuple[datasets.ImageFolder, List[str]]: The dataset and the list of classes.
    """
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    classes = dataset.classes
    return dataset, classes


def split_dataset(
    dataset: datasets.ImageFolder, train_split: float = 0.7, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """Loads the dataset and splits it into train and test sets.

    Args:
        dataset (str): The dataset to split.
        train_split (float, optional): Percentage for trainning. Defaults to 0.7.
        batch_size (int, optional): Size of batching. Defaults to 32.

    Returns:
        Tuple[DataLoader, DataLoader]: The train and test DataLoader objects.
    """
    num_workers = 2
    dataset_len = len(dataset)

    indices = torch.randperm(dataset_len).tolist()
    split = int(np.floor(train_split * dataset_len))
    train_indices, test_indices = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset,
        sampler=test_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return train_loader, test_loader
