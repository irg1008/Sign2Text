import torch


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
        for i, (videos, targets) in enumerate(loader):
            videos, targets = videos.to(device), targets.to(device)

            scores = model(videos)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

            print(f"Predictions for batch {i+1} ")
            print([classes[int(i)] for i in predictions])

            print(f"Ground truth for batch {i+1}")
            print([classes[int(i)] for i in targets])

            print("---------------------------------\n\n")
            break

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
