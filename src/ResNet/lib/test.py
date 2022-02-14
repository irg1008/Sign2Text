import torch
from utils.output import imshow
from torchvision.utils import make_grid


def check_accuracy(loader, model, classes, device):
    """_summary_

    Args:
        loader (_type_): _description_
        model (_type_): _description_
        classes (_type_): _description_
    """
    model.to(device)
    model.eval()

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images, targets = images.to(device), targets.to(device)

            scores = model(images)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

            # Output both images to compare.
            print("Images grid")
            imshow(make_grid(images.cpu()))

            print(f"Predictions for batch {i+1} ")
            print([classes[int(i)] for i in predictions])

            print(f"Ground truth for batch {i+1}")
            print([classes[int(i)] for i in targets])

            print("---------------------------------\n\n")
            break

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )


def predict_class(loader, model, classes, debug_class, device):
    """_summary_

    Args:
        loader (_type_): _description_
        model (_type_): _description_
        classes (_type_): _description_
        debug_class (_type_): _description_
    """
    model.to(device)
    model.eval()

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)

        scores = model(images)
        _, predictions = scores.max(1)

        for i, (image, target) in enumerate(zip(images, targets)):

            label = classes[target]
            if label != debug_class:
                continue

            # Predict label for image.
            prediction = classes[predictions[i]]

            # Show image.
            imshow(image.cpu())

            print(f"Prediction: {prediction}. Ground truth: {label}")

            return
