import torch
from typing import List
from torchvision import transforms
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImgsToTensor(torch.nn.Module):
    @staticmethod
    def forward(img_list: List[Image.Image]) -> torch.Tensor:
        """Converts a list of PIL images to a tensor.

        Args:
            img_list (List[Image.Image]): The list of images to convert.

        Returns:
            torch.Tensor: The tensor of the images.
        """
        return torch.stack([transforms.ToTensor()(img) for img in img_list]).squeeze(
            dim=1
        )


# Transform for multiple images.
get_transform = lambda image_size: transforms.Compose(
    [
        ImgsToTensor(),
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

reverse_transform = transforms.Compose(
    [
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
        ),
    ]
)