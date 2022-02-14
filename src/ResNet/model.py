from torchvision.models import ResNet, resnet50
from torch import nn, optim
from config import learning_rate
import torch


def create_model() -> ResNet:
    # Load custom model.
    # TODO

    # Load pretrain model & modify it.
    # model = models.resnet18(pretrained=True)
    model = resnet50(pretrained=True)

    return model


# Change last layer.
def change_last_layer(model: ResNet, num_classes: int):
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    return criterion, optimizer


def export_model(model, path: str):
    torch.save(model, path)


def load_model(model_path: str):
    model = torch.load(model_path)
    return model
