from torch import nn, optim


def optim_model(model, learning_rate: float):
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Useful link for scheduler: https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", min_lr=1e-6, factor=0.7, patience=5
    )

    return criterion, optimizer, scheduler


def get_correct(scores, targets):
    """_summary_

    Args:
        scores (_type_): _description_
        targets (_type_): _description_

    Returns:
        _type_: _description_
    """
    _, predictions = scores.max(1)
    acc = (predictions == targets).sum()
    num = predictions.size(0)
    return acc, num


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    learning_rate,
    num_epochs,
):
    """_summary_

    Args:
        model (_type_): _description_
        train_loader (_type_): _description_

    Returns:
        _type_: _description_
    """
    criterion, optimizer, scheduler = optim_model(model, learning_rate)
    model.to(device)

    print(f"Training on device: {device}")

    costs, val_costs, accs, val_accs = [], [], [], []

    # Train network.
    model.train()
    for epoch in range(num_epochs):
        train_losses, val_losses = [], []
        train_correct, val_correct = 0, 0
        train_predictions, val_predictions = 0, 0

        model.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()

            # forward.
            scores = model(data)
            loss = criterion(scores, targets)
            train_losses.append(loss.item())

            # Save acc.
            correct, num = get_correct(scores, targets)
            train_correct += correct
            train_predictions += num

            # backward.
            loss.backward()

            # gradient descent.
            optimizer.step()

        model.eval()
        for data, targets in validation_loader:
            data, targets = data.to(device), targets.to(device)

            # forward.
            scores = model(data)
            loss = criterion(scores, targets)
            val_losses.append(loss.item())

            # Save acc.
            correct, num = get_correct(scores, targets)
            val_correct += correct
            val_predictions += num

        cost = sum(train_losses) / len(train_losses)
        costs.append(cost)
        acc = 100.0 * float(train_correct) / float(train_predictions)
        accs.append(acc)

        val_cost = sum(val_losses) / len(val_losses)
        val_costs.append(val_cost)
        val_acc = 100.0 * float(val_correct) / float(val_predictions)
        val_accs.append(val_acc)

        scheduler.step(val_cost)

        # Print load bar for epoch with cost and acc info.
        print(
            f"Epoch: {epoch + 1}/{num_epochs} | "
            f"Train Loss: {cost:.4f} | "
            f"Val Loss: {val_cost:.4f} | "
            f"Train Acc: {acc:.4f}% | "
            f"Val Acc: {val_acc:.4f}% | ",
            f"LR: {optimizer.param_groups[0]['lr']}",
        )

    return costs, val_costs, accs, val_accs
