from .video_dataset import VideoFrameDataset
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def split_dataset(
    dataset: VideoFrameDataset, train_split: float = 0.7, batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
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
