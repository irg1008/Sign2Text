import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import UploadFile
from torchvision import io
from torch import Tensor
from torchvision import transforms
import numpy as np

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_frames(path: str) -> Tensor:
    """Get frames from video.

    Args:
        path (str): Path to video.

    Returns:
        Tensor: Frames from video.
    """
    frames = io.read_video(path)[0]
    return frames.float()


async def create_tmp_path(file: UploadFile) -> str:
    """Create a temporary path for the file

    Args:
        file (UploadFile): File to save.

    Returns:
        str: Path to the temporary file.
    """
    try:
        suffix = Path(file.filename).suffix
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    finally:
        await file.close()
    return tmp_path


async def process_video(frames: Tensor) -> Tensor:
    """Process video.

    Args:
        frames (Tensor): Video to process.

    Returns:
        Tensor: Frames from video.
    """

    # 1. Reorder RGB channels to position 3.
    # (fps, image size, image size, channels) -> (fps, channels, image size, image size)
    frames = frames.permute(0, 3, 1, 2)

    # 2. Resize frames. From WxH to 224x224.
    frames = transform(frames)

    # 3. Get n frames.
    n_frames = 50
    if len(frames > 50):
        # a. Equally separated.
        idx = np.round(np.linspace(0, len(frames) - 1, n_frames)).astype(int)
        frames = frames[idx]
    else:
        # b. Duplicate until 50 frames.
        frames = frames.repeat(np.ceil(n_frames / len(frames)), 1, 1, 1)[:n_frames]

    # 4. Add batch dimension.
    frames = frames.unsqueeze(0)

    return frames
