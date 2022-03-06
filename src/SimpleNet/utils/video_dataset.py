from matplotlib import pyplot as plt
from os import listdir, path
from torch.utils import data
from torch import arange
from typing import List
from torchvision.transforms import Compose
from PIL import Image
import numpy as np

# Based on: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch/blob/main/video_dataset.py


class VideoRecord(object):
    def __init__(self, path: str, label: int):
        self._path = path
        self._label = label
        self._num_frames = len(listdir(path))

    @property
    def path(self):
        return self._path

    @property
    def label(self):
        return self._label

    @property
    def num_frames(self):
        return self._num_frames


class VideoFrameDataset(data.Dataset):
    def __init__(
        self,
        root_path: str,
        transform: Compose,
        num_segments: int,
        frames_per_segment: int,
        imagefile_template: str = "img_{:05d}.png",
    ):
        super(VideoFrameDataset, self).__init__()
        self.root_path = root_path
        self.transform = transform
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template

        self._load_data()

    def _load_data(self):
        self.videos: List[VideoRecord] = list()
        self.classes = listdir(self.root_path)
        self.targets = list(range(len(self.classes)))

        for target, label in zip(self.targets, self.classes):
            videos_path = path.join(self.root_path, label)

            for video in listdir(videos_path):
                video_record = VideoRecord(
                    path.join(self.root_path, videos_path, video), target
                )
                self.videos.append(video_record)

    def _load_image(self, directory: str, frame: int) -> Image.Image:
        return Image.open(
            path.join(directory, self.imagefile_template.format(frame + 1))
        ).convert("RGB")

    def _get_start_indices(self, record: VideoRecord):
        return np.linspace(
            0, record.num_frames - self.frames_per_segment, self.num_segments, dtype=int
        )

    def _get(self, record: VideoRecord, frame_start_indices: np.ndarray):
        images = list()

        for start_index in frame_start_indices:
            for i in range(self.frames_per_segment):
                if start_index + i >= record.num_frames:
                    break
                image = self._load_image(record.path, start_index + i)
                images.append(image)

        images_tensor = self.transform(images)
        return images_tensor, record.label

    def __getitem__(self, idx: int):
        record = self.videos[idx]
        frame_start_indices = self._get_start_indices(record)
        return self._get(record, frame_start_indices)

    def __len__(self):
        return len(self.videos)
