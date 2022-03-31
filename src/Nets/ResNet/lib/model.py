from torchvision.models import ResNet, resnet18
from torch import nn, save, load


def create_model(num_classes: int) -> ResNet:
    """_summary_

    Returns:
        ResNet: _description_
    """
    # Load custom model. Hacer

    # Load pretrain model & modify it.
    # model = models.resnet18(pretrained=True)
    model = resnet18(pretrained=True)
    update_last_layer(model, num_classes)
    return model


# Change last layer.
def update_last_layer(model: ResNet, num_classes: int) -> ResNet:
    """_summary_

    Args:
        model (ResNet): _description_
        num_classes (int): _description_
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
    """_summary_

    Args:
        model (_type_): _description_
        path (str): _description_
    """
    save(model, path)


def load_model(model_path: str):
    """_summary_

    Args:
        model_path (str): _description_

    Returns:
        _type_: _description_
    """
    model = load(model_path)
    return model
