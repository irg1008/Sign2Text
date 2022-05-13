from torch import nn, optim


def optim_model(model, learning_rate: float):
    """_summary_

    Args:
        model (_type_): _description_

    Returns:
        _type_: _description_
    """
    class_criterion = nn.CrossEntropyLoss()
    pose_criterion = nn.MSELoss()

    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Useful link for scheduler: https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", min_lr=1e-6, factor=0.7, patience=10
    )

    return class_criterion, pose_criterion, optimizer, scheduler


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


def net_pass(model, data, criterion_1, criterion_2, target_1, target_2):
    """Get loss for model.

    Args:
        model (_type_): _description_
        criterion (_type_): _description_
        data (_type_): _description_
        target_1 (_type_): _description_
        target_2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    out_1, out_2 = model(data)
    loss_1, loss_2 = criterion_1(out_1, target_1), criterion_2(out_2, target_2)
    loss = loss_1
    loss += loss_2
    acc, num = get_correct(out_1, target_1)
    return loss, acc, num


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
    class_criterion, pose_criterion, optimizer, scheduler = optim_model(
        model, learning_rate
    )
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
        for data, (class_targets, pose_targets) in train_loader:
            data, class_targets, pose_targets = (
                data.to(device),
                class_targets.to(device),
                pose_targets.to(device),
            )

            optimizer.zero_grad()

            # forward.
            loss, acc, num = net_pass(
                model,
                data,
                class_criterion,
                pose_criterion,
                class_targets,
                pose_targets,
            )
            train_losses.append(loss.item())

            # Save acc.
            train_correct += acc
            train_predictions += num

            # backward.
            loss.backward()

            # gradient descent.
            optimizer.step()

        model.eval()
        for data, (class_targets, pose_targets) in validation_loader:
            data, class_targets, pose_targets = (
                data.to(device),
                class_targets.to(device),
                pose_targets.to(device),
            )

            # forward.
            loss, acc, num = net_pass(
                model,
                data,
                class_criterion,
                pose_criterion,
                class_targets,
                pose_targets,
            )
            val_losses.append(loss.item())

            # Save acc.
            val_correct += acc
            val_predictions += num

        # Summarize acc and loss.
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
