import torch
from torchvision.models import ResNet


def quantize_model(model: ResNet) -> ResNet:
    """Quantize the model.

    Args:
        model (ResNet): The model to quantize.

    Returns:
        ResNet: The quantized model.
    """
    cpu_model = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        cpu_model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model
