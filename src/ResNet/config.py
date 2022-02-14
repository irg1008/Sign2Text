from typing import Tuple
from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
NUM_EPOCHS = 25
TRAIN_SPLIT = 0.7
IMAGE_SIZE = (224, 224)
NUM_WORKERS = 2

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_path_and_names() -> Tuple[str, str, str, str]:
    """Get the path to the dataset and the names of the classes.

    Returns:
        Tuple[str, str, str, str]: The path to the dataset, the path to the train set, the path to the test set and the list of classes.
    """
    datasets_info = {
        "WLASL": {"name": "WLASL_frames_100", "path": "WLASL/frames_100"},
        "animals": {"name": "animals_simple", "path": "animals/all"},
    }

    dataset_name = "animals"  # Change this.
    debug_class = "dog"  # Change this.

    model_name = datasets_info[dataset_name]["name"]

    data_dir = f"../../data/{datasets_info[dataset_name]['path']}"
    model_path = f"../../models/resnet_{model_name}.pth"
    quant_model_path = f"../../models/resnet_{model_name}_quantized.pth"

    return data_dir, model_path, quant_model_path, debug_class
