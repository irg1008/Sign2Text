import sys

sys.path.append("../")


from config.dataset import get_dataset_path
from config.torch_config import get_transform
from utils.loader import split_dataset

from lib.video_dataset import VideoFrameDataset

if __name__ == "__main__":
    data_path, model_path = get_dataset_path(
        dataset="WLASL/videos", model_name="WLASL_5"
    )
    multiple_transform = get_transform(200)

    dataset = VideoFrameDataset(
        root_path=data_path,
        transform=multiple_transform,
        image_size=200,
        num_segments=2,
        frames_per_segment=18,
    )
    classes = dataset.classes

    train_loader, test_loader, validation_loader = split_dataset(
        dataset, train_split=0.70, validation_split=0.1, batch_size=16
    )

    inputs, (classes, poses) = next(iter(train_loader))
    print(
        poses.shape
    )  # (16, 67, 20). 67 keypoints with xy data (20 = 2 of xy * 10 frames).
    print(inputs.shape)  # (16, 10, 3, 180, 180).
    print(classes.shape)  # (16)
