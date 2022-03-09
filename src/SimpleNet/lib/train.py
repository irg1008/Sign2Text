from torch import nn, optim, squeeze


def optim_model(model, learning_rate: float):
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    return criterion, optimizer


def train_model(model, train_loader, device, learning_rate, num_epochs):
    """_summary_

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_

    Returns:
        _type_: _description_
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
