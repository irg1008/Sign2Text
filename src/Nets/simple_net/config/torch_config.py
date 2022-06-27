import sys
from torchvision import transforms

sys.path.append("../")
from common.config.torch_config import ImgsToTensor, normalize


def get_transform(image_size: int, random_resize: float):
    """Get the transform for the dataset.

    Args:
        image_size (int): The image size.
        random_resize (float): The random resize scale (0 to 1).

    Returns:
        transforms.Compose: The transform.
    """
    return transforms.Compose(
        [
            ImgsToTensor(),
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size * random_resize),
            normalize,
        ]
    )
