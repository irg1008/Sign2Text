from os import listdir, path
from typing import List
from torch.utils import data
from torchvision.transforms import Compose
from PIL import Image
from numpy import ndarray
import numpy as np

# Based on: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch/blob/main/video_dataset.py


def load_image(directory: str, frame: int, imagefile_template: str) -> Image.Image:
    """Load an image from a directory and frame.

    Args:
        directory (str): The directory of the video.
        frame (int): The frame number.

    Returns:
        Image.Image: The image loaded from directory.
    """
    return Image.open(
        path.join(directory, imagefile_template.format(frame + 1))
    ).convert("RGB")


class VideoRecord:
    """Video record to store video data."""

    def __init__(self, path: str, label: int):
        self._path = path
        self._label = label
        self._num_frames = len([f for f in listdir(path) if f.endswith(".png")])

    @property
    def path(self) -> str:
        """Get the video path

        Returns:
            str: The video path.
        """
        return self._path

    @property
    def label(self) -> int:
        """Get the video label

        Returns:
            int:  The video label.
        """
        return self._label

    @property
    def num_frames(self) -> int:
        """Get the number of frames in the video.

        Returns:
            int: The number of frames in the video.
        """
        return self._num_frames


class VideoFrameDataset(data.Dataset):
    """Dataset that loads videos from a directory of images"""

    def __init__(
        self,
        root_path: str,
        transform: Compose,
        num_segments: int,
        frames_per_segment: int,
        imagefile_template: str = "img_{:05d}.png",
    ):
        super().__init__()
        self.transform = transform
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self._load_data(root_path)

    def _load_data(self, root_path: str):
        """Load the video data."""
        self.videos: List[VideoRecord] = []
        self.classes = listdir(root_path)
        self.targets = list(range(len(self.classes)))

        for target, label in zip(self.targets, self.classes):
            videos_path = path.join(root_path, label)

            for video in listdir(videos_path):
                video_record = VideoRecord(
                    path.join(root_path, videos_path, video), target
                )
                self.videos.append(video_record)

    def _get_start_indices(self, record: VideoRecord) -> ndarray:
        """Get the start indices for the video.

        Args:
            record (VideoRecord): The video record.

        Returns:
            ndarray: The start indices for the video.
        """
        return np.linspace(
            0, record.num_frames - self.frames_per_segment, self.num_segments, dtype=int
        )

    def _get(self, record: VideoRecord, frame_start_indices: np.ndarray):
        """Get the next item on dataset.

        Args:
            record (VideoRecord): The video record.
            frame_start_indices (np.ndarray): The start indices for the video.

        Returns:
            Tuple[Tensor, int]: The next item on dataset.
        """
        images = []

        for start_index in frame_start_indices:
            for i in range(self.frames_per_segment):
                if start_index + i >= record.num_frames:
                    break
                image = load_image(
                    record.path, start_index + i, self.imagefile_template
                )
                images.append(image)

        images_tensor = self.transform(images)
        return images_tensor, record.label

    def __getitem__(self, idx: int):
        """Get the next item on dataset.

        Args:
            idx (int): The index of the video.

        Returns:
            Tuple[Tensor, int]: The next item on dataset.
        """
        record = self.videos[idx]
        frame_start_indices = self._get_start_indices(record)
        return self._get(record, frame_start_indices)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.videos)
