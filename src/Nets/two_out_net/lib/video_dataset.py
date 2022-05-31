from os import listdir, path
from typing import List, Tuple
from torch import Tensor
from torch.utils import data
from torchvision.transforms import Compose
from PIL import Image
from numpy import ndarray
import numpy as np
import json
import torch

# Based on: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch/blob/main/video_dataset.py


class Pose:
    """Pose class."""

    def __init__(
        self,
        pose_keypoints_2d: List[int] = np.zeros(75).tolist(),
        face_keypoints_2d: List[int] = np.zeros(0).tolist(),
        hand_left_keypoints_2d: List[int] = np.zeros(63).tolist(),
        hand_right_keypoints_2d: List[int] = np.zeros(63).tolist(),
    ):
        self.body_pose = pose_keypoints_2d
        self.face_pos = face_keypoints_2d
        self.left_hand_pos = hand_left_keypoints_2d
        self.right_hand_pos = hand_right_keypoints_2d

    def to_tensor(self):
        """Converts the pose to a nd array."""

        pose: List[int] = []
        # pose += self.body_pose
        pose += self.face_pos
        pose += self.left_hand_pos
        pose += self.right_hand_pos

        # Normalize between 0 and 1.
        original_width = 256
        norm = lambda x: x / original_width
        get_every_row = lambda n: np.array(
            [v for i, v in enumerate(pose) if i % 3 == n]
        )

        x = torch.FloatTensor(norm(get_every_row(0)))
        y = torch.FloatTensor(norm(get_every_row(1)))

        xy = torch.stack([x, y]).transpose(0, 1)
        return xy


def load_pose(directory: str, frame: int, posefile_template: str, img_width: int):
    """Load the pose from a directory and frame.

    Args:
        directory (str): The directory of the video.
        frame (int): The frame number.

    Returns:
        Pose: The pose loaded from directory.
    """
    f = open(
        path.join(directory, posefile_template.format(frame + 1)),
        encoding="utf-8",
    )
    pose_json = json.load(f, parse_int=int)

    people = pose_json["people"]
    if len(people) == 0:
        return None

    person = pose_json["people"][0]
    pose = Pose(
        pose_keypoints_2d=person["pose_keypoints_2d"],
        face_keypoints_2d=person["face_keypoints_2d"],
        hand_left_keypoints_2d=person["hand_left_keypoints_2d"],
        hand_right_keypoints_2d=person["hand_right_keypoints_2d"],
    )

    return pose


def get_poses_tensor(poses: List[Pose]) -> Tensor:
    # return torch.cat([pose.to_tensor() for pose in poses]).flatten()
    return torch.stack([pose.to_tensor() for pose in poses])


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
    """Vdieo record to store video data."""

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


Input = Tensor
Target = Tuple[int, Tensor]


class VideoFrameDataset(data.Dataset):
    """Dataset that loads videos from a directory of images"""

    def __init__(
        self,
        root_path: str,
        transform: Compose,
        image_size: int,
        num_segments: int,
        frames_per_segment: int,
        imagefile_template: str = "img_{:05d}.png",
        posefile_template: str = "img_{:05d}.json",
    ):
        super().__init__()
        self.transform = transform
        self.image_size = image_size
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template
        self.posefile_template = posefile_template
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
        # if frame start indices contains < 0
        stop = np.max([0, record.num_frames - self.frames_per_segment])
        return np.linspace(0, stop, self.num_segments, dtype=int)

    def _get(
        self, record: VideoRecord, frame_start_indices: np.ndarray
    ) -> Tuple[Tensor, int, Tensor]:
        """Get the next item on dataset.

        Args:
            record (VideoRecord): The video record.
            frame_start_indices (np.ndarray): The start indices for the video.

        Returns:
            Tuple[Tensor, int]: The next item on dataset.
        """
        images = []
        poses: List[Pose] = []

        for start_index in frame_start_indices:
            for i in range(start_index, start_index + self.frames_per_segment):
                if i >= record.num_frames:
                    break
                image = load_image(record.path, i, self.imagefile_template)
                pose = load_pose(
                    record.path, i, self.posefile_template, self.image_size
                )
                if pose is None:
                    pose = poses[-1] if len(poses) > 0 else Pose()
                images.append(image)
                poses.append(pose)

        images_tensor = self.transform(images)
        poses_tensor = get_poses_tensor(poses)
        return images_tensor, record.label, poses_tensor

    def __getitem__(self, idx: int) -> Tuple[Input, Target]:
        """Get the next item on dataset.

        Args:
            idx (int): The index of the video.

        Returns:
            Tuple[Tensor, int]: The next item on dataset.
        """
        record = self.videos[idx]
        frame_start_indices = self._get_start_indices(record)
        video, label, poses = self._get(record, frame_start_indices)

        return video, (label, poses)

    def __len__(self):
        """Get the length of the dataset."""
        return len(self.videos)
