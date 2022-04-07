from utils.output import imshow


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
