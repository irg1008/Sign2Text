from torchvision.models import ResNet, resnet50
from torch import nn, optim
from config import LEARNING_RATE
import torch


def create_model() -> ResNet:
    """_summary_

    Returns:
        ResNet: _description_
    """
    # Load custom model. Hacer

    # Load pretrain model & modify it.
    # model = models.resnet18(pretrained=True)
    model = resnet50(pretrained=True)

    return model


# Change last layer.
def change_last_layer(model: ResNet, num_classes: int):
    """_summary_

    Args:
        model (ResNet): _description_
        num_classes (int): _description_
    """
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes),
        nn.LogSoftmax(dim=1),
    )

    # model.fc = nn.Linear(2048, num_classes)


def optim_model(model):
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

    return criterion, optimizer


def export_model(model, path: str):
    """_summary_

    Args:
        model (_type_): _description_
        path (str): _description_
    """
    torch.save(model, path)


def load_model(model_path: str):
    """_summary_

    Args:
        model_path (str): _description_

    Returns:
        _type_: _description_
    """
    model = torch.load(model_path)
    return model
