from .video_dataset import VideoFrameDataset
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def split_dataset(
    dataset: VideoFrameDataset,
    train_split: float = 0.7,
    validation_split: float = 0.1,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    num_workers = 2
    dataset_len = len(dataset)

    indices = torch.randperm(dataset_len).tolist()
    split = int(np.floor(train_split * dataset_len))
    second_split = int(np.floor((train_split + validation_split) * dataset_len))
    
    train_indices, validation_indices, test_indices = (
        indices[:split],
        indices[split:second_split],
        indices[second_split:],
    )

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)

    get_loader = lambda sampler: DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers
    )

    train_loader = get_loader(train_sampler)
    validation_loader = get_loader(validation_sampler)
    test_loader = get_loader(test_sampler)

    return train_loader, test_loader, validation_loader
