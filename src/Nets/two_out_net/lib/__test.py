import sys
import torch

sys.path.append("../")


from config.dataset import get_dataset_path
from common.config.torch_config import get_transform
from utils.loader import split_dataset

from lib.video_dataset import VideoFrameDataset
from lib.model import CNN


def test_dataset():
    """Test the dataset."""
    data_path, _ = get_dataset_path(dataset="WLASL/videos", model_name="WLASL_5")
    multiple_transform = get_transform(200)

    dataset = VideoFrameDataset(
        root_path=data_path,
        transform=multiple_transform,
        image_size=200,
        num_segments=2,
        frames_per_segment=18,
    )
    classes = dataset.classes

    train_loader, _, _ = split_dataset(
        dataset, train_split=0.70, validation_split=0.1, batch_size=16
    )

    inputs, (classes, poses) = next(iter(train_loader))
    print(
        poses.shape
    )  # (16, 67, 20). 67 keypoints with xy data (20 = 2 of xy * 10 frames).
    print(inputs.shape)  # (16, 10, 3, 180, 180).
    print(classes.shape)  # (16)


def test_model():
    """Test the model."""
    num_frames = 10 * 5
    model = CNN(
        num_classes=5,
        num_frames=num_frames,
        image_size=112,
        num_pose_points=42 * 2 * num_frames,
    )

    model.forward(torch.randn(8, 50, 3, 112, 112))


if __name__ == "__main__":
    test_model()
