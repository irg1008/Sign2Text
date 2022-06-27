from torch import nn, optim


def optim_model(model, learning_rate: float):
    """Get critetric and optimizer for model. Additionally, get scheduler.

    Args:
        model (Model): A model for optimizer = [Example = torch.optim.Adadelta(net.parameters(), lr=1e-2)].

    Returns:
        criterion (nn.CrossEntropyLoss): A criterion for loss.
        optimizer (torch.optim.Optimizer): An optimizer for model.
        scheduler (torch.optim.lr_scheduler.ReduceLROnPlateau): A scheduler for learning rate.
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
    """
    Get correct predictions and total predictions.

    Args:
        scores (torch.Tensor): A tensor of scores.
        targets (torch.Tensor): A tensor of targets.

    Returns:
        correct (int): Number of correct predictions.
        num (int): Number of total predictions.
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
    """
    Train a model.

    Args:
        model (Model): A model to train.
        train_loader (DataLoader): A DataLoader object.
        validation_loader (DataLoader): A DataLoader object.
        device (str): A device to train on.
        learning_rate (float): A learning rate.
        num_epochs (int): Number of epochs to train.

    Returns:
        model (Model): A trained model.
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
