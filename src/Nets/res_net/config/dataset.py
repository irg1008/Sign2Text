from typing import Dict, Tuple
from os import path


def get_datasets() -> Dict[str, Dict[str, str]]:
    """Get list of available local datasets.

    Returns:
        Dict[str, Dict[str, str]]: The list of available datasets.
    """
    datasets = {
        "WLASL": {"name": "WLASL", "path": "WLASL/frames"},
        "animals": {"name": "animals", "path": "animals/all"},
        "signs": {"name": "signs", "path": "signs"},
    }
    return datasets


def get_dataset_info(dataset: str) -> Tuple[str, str]:
    """Get the path for dataset data and output model path.

    Args:
        dataset (Literal[WLASL, animals]): The dataset name.

    Returns:
        Tuple[str, str]: The path to the dataset and the path to the output model.
    """
    datasets = get_datasets()
    name, data_path = datasets[dataset].values()

    # Base path for project.
    base_path = path.join(path.dirname(__file__), "../../../../")

    data_dir = path.abspath(path.join(base_path, "data", data_path))
    model_path = path.abspath(path.join(base_path, "models", f"{name}_resnet.pth"))

    return data_dir, model_path
