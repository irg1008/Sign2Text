from os import path
import os
from typing import Tuple


def get_dataset_path() -> Tuple[str, str]:
    base_path = path.join(path.dirname(__file__), "../../../")
    data_path = path.abspath(path.join(base_path, "data", "WLASL", "videos"))
    model_path = path.abspath(
        path.join(base_path, "models", f"resnet_WLASL_videos.pth")
    )

    return data_path, model_path
