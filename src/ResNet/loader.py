from typing import List, Tuple
from torch.utils.data.sampler import SubsetRandomSampler
from config import transform, num_workers
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchvision import datasets


def load_split_dataset(
    data_dir: str, train_split: float = 0.7, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Loads the dataset and splits it into train and test sets.

    Args:
        data_dir (str): The path to the dataset.
        train_split (float, optional): Percentage for trainning. Defaults to 0.7.
        batch_size (int, optional): Size of batching. Defaults to 32.

    Returns:
        Tuple[DataLoader, DataLoader, List[str]]: The train and test DataLoader objects and the list of classes.
    """

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    classes = dataset.classes
    dataset_len = len(dataset)

    ran_ind = torch.randperm(dataset_len).tolist()  # 1. random
    # seq_ind = list(range(dataset_len))  # 2. sequential

    indices = ran_ind
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

    return train_loader, test_loader, classes
