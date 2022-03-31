import torch
from torchvision import transforms

# Get available device for torch. Cuda | CPU.
get_device = lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get tranform object based on image size.
get_transform = lambda image_size, width_multiplier: transforms.Compose(
    [
        transforms.Resize((image_size, image_size * width_multiplier)),
        # transforms.RandomHorizontalFlip(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),
    ]
)
