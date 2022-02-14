import torch
from torch import nn


def quantize_model(model):
    cpu_model = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        cpu_model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model
