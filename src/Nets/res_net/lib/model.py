from torchvision.models import ResNet, resnet18
from torch import nn, save, load


def create_model(num_classes: int) -> ResNet:
    """Create a ResNet model.

    Returns:
        ResNet: A ResNet model.
    """
    # Load custom model. Hacer

    # Load pretrain model & modify it.
    # model = models.resnet18(pretrained=True)
    model = resnet18(pretrained=True)
    update_last_layer(model, num_classes)
    return model


# Change last layer.
def update_last_layer(model: ResNet, num_classes: int) -> ResNet:
    """Update last layer of a ResNet model.

    Args:
        model (ResNet): A ResNet model.
        num_classes (int): Number of classes.
    """
    for param in model.parameters():
        param.requires_grad = False

    # model.fc = nn.Sequential(
    #     nn.Linear(2048, 512),
    #     nn.ReLU(),
    #     nn.Dropout(0.2),
    #     nn.Linear(512, num_classes),
    #     nn.LogSoftmax(dim=1),
    # )

    model.fc = nn.Linear(512, num_classes)
    return model


def export_model(model, path: str):
    """Export a model to a file.

    Args:
        model (ResNet): A ResNet model to export.
        path (str): Path to save the model to.
    """
    save(model, path)


def load_model(model_path: str):
    """Load a model from a given path.

    Args:
        model_path (str): Path to load the model from.

    Returns:
        ResNet: A ResNet model.
    """
    model = load(model_path)
    return model
