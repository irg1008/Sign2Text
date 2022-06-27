from os import path
from typing import Tuple


def get_dataset_path(dataset: str, model_name: str) -> Tuple[str, str]:
    """Get the dataset path.

    Args:
        dataset (str): The dataset name.
        model_name (str): The model name.

    Returns:
        Tuple[str, str]: The dataset path and the model path.
    """
    base_path = path.join(path.dirname(__file__), "../../../../")
    data_path = path.abspath(path.join(base_path, "data", dataset))
    model_path = path.abspath(path.join(base_path, "models", f"{model_name}_twonet.pth"))
    return data_path, model_path
