import sys
from torchvision import transforms

sys.path.append("../")
from common.config.torch_config import ImgsToTensor, normalize


# Transform for multiple images.
get_transform = lambda image_size, random_resize: transforms.Compose(
    [
        ImgsToTensor(),
        transforms.Resize(image_size),
        transforms.RandomCrop(image_size * random_resize),
        normalize,
    ]
)
