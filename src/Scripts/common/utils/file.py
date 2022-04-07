from os import path


def abs_path(custom_path: str) -> str:
    """Returns the absolute path of a relative path.

    Args:
        custom_path (str): path to check.

    Returns:
        str: absolute path.
    """
    # Check if already absolute.
    is_absolute = path.isabs(custom_path)
    if is_absolute:
        return custom_path

    # If not absolute, use relative to current directoy.
    return path.abspath(
        path.join(
            path.abspath(path.dirname(__file__)),
            custom_path,
        )
    )
