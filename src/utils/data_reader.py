import json
from os import path
from typing import Dict, List


labels: Dict[str, List[int]] = {}


def load_labels() -> None:
    """Load all labels from WLASL json file."""
    file_path = "../../data/WLASL.json"
    file_path = path.abspath(
        path.join(
            path.abspath(path.dirname(__file__)),
            file_path,
        )
    )

    with open(file_path, encoding="utf8") as ipf:
        content = json.load(ipf)

    for entry in content:
        label = entry["gloss"]
        labels[label] = []

        for video in entry["instances"]:
            labels[label].append(video["video_id"])


def get_label_from_id(
    video_id: str,
) -> str:
    """Get label from video id.

    Args:
        video_id (int): Video id.

    Raises:
        ValueError: If video id is not found.

    Returns:
        str: Label.
    """
    if not labels:
        load_labels()

    for label, ids in labels.items():
        if video_id in ids:
            return label
    raise ValueError("Video ID not found")


if __name__ == "__main__":
    load_labels()
    print(get_label_from_id("35463"))
