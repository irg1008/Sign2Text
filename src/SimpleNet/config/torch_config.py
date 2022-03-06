import torch

# Create unified tensor for concatenation.
get_concatenated_transform = lambda images: torch.stack(images, dim=0)
