from typing import List, Tuple


def check_balance_status(loader, classes) -> List[Tuple[str, int]]:
    """Checks the balance status of the dataset.

    Args:
        loader ([type]): Loader to check.
        classes ([type]): List of classes.

    Returns:
        List[Tuple[str, int]]: [description]
    """
    class_count = {}

    for _, targets in loader:
        for target in targets:
            label = classes[target]
            if label not in class_count:
                class_count[label] = 0
            class_count[label] += 1

    info = sorted(class_count.items())
    return info
