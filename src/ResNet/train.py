from config import NUM_EPOCHS, device
from model import optim_model


def train_model(model, train_loader):
    """_summary_

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_

    Returns:
        _type_: _description_
    """
    model.to(device)
    model.train()

    criterion, optimizer = optim_model(model)

    print(f"Training on device: {device}")

    # Train network.
    costs = []

    for epoch in range(NUM_EPOCHS):
        losses = []

        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            # forward.
            scores = model(data)
            loss = criterion(scores, targets)
            losses.append(loss.item())

            # backward.
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step.
            optimizer.step()

        cost = sum(losses) / len(losses)
        costs.append(cost)
        print(f"Cost at epoch {epoch + 1} is {cost:.5f}")

    return costs
