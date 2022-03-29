from os import path
import os
from typing import Tuple, Literal


def get_dataset_path(
    dataset: Literal["WLASL/videos", "signs"] = "WLASL/videos", model_name="WLASL"
) -> Tuple[str, str]:
    base_path = path.join(path.dirname(__file__), "../../../")
    data_path = path.abspath(path.join(base_path, "data", dataset))
    model_path = path.abspath(path.join(base_path, "models", f"{model_name}.pth"))
    return data_path, model_path
