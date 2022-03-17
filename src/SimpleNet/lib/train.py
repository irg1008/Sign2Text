from torch import nn, optim, squeeze


def optim_model(model, learning_rate: float):
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    return criterion, optimizer


def train_model(
    model, train_loader, validation_loader, device, learning_rate, num_epochs
):
    """_summary_

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_

    Returns:
        _type_: _description_
    """
    criterion, optimizer = optim_model(model, learning_rate)
    model.to(device)

    print(f"Training on device: {device}")

    # Train network.
    costs = []
    val_costs = []

    model.train()
    for epoch in range(num_epochs):
        test_losses = []
        val_losses = []

        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            # forward.
            scores = model(data)
            loss = criterion(scores, targets)
            test_losses.append(loss.item())

            # backward.
            optimizer.zero_grad()
            loss.backward()

            # gradient descent or adam step.
            optimizer.step()

        model.eval()
        for data, targets in validation_loader:
            data, targets = data.to(device), targets.to(device)

            # forward.
            scores = model(data)
            loss = criterion(scores, targets)
            val_losses.append(loss.item())

        cost = sum(test_losses) / len(test_losses)
        costs.append(cost)

        val_cost = sum(val_losses) / len(val_losses)
        val_costs.append(val_cost)

        print(f"Train cost at epoch {epoch + 1} is {cost:.5f}")
        print(f"Validation cost at epoch {epoch + 1} is {val_cost:.5f}")

    return costs
