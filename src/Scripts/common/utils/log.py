from typing import Literal


def log(
    msg: str,
    mode: Literal[
        "info",
        "warning",
    ] = "info",
    delete_previous=False,
    double_new_line=False,
) -> None:
    """Log msg.

    Args:
        msg (str): msg to log.
        type (Literal[, optional): type of msg. Defaults to "Info".
        delete_previous (bool, optional): delete previous log. Defaults to False.
    """
    print(
        "\n" if double_new_line else "",
        f"[{mode}] - Video2Frame: {msg}",
        end="\r" if delete_previous else "",
    )
