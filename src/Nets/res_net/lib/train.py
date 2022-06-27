from torch import nn, optim


def optim_model(model, learning_rate: float):
    """Get creterion and optimizer for training phase.

    Args:
        model (Model): A model to train.

    Returns:
        criterion (nn.CrossEntropyLoss): A criterion for loss.
        optimizer (Optimizer): An optimizer for training.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    return criterion, optimizer


def train_model(model, train_loader, device, learning_rate, num_epochs):
    """Train a model.

    Args:
        model (Model): A model to train.
        train_loader (DataLoader): A DataLoader object.

    Returns:
        model (Model): A trained model.
    """
    criterion, optimizer = optim_model(model, learning_rate)

    model.to(device)
    model.train()

    print(f"Training on device: {device}")

    # Train network.
    costs = []

    for epoch in range(num_epochs):
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
