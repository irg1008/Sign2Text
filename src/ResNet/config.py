from torchvision import transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

learning_rate = 1e-3
batch_size = 128
num_epochs = 25
train_split = 0.7
image_size = (224, 224)
num_workers = 2

transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_path_and_names():
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
