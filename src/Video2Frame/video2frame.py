import json
from typing import Dict, Literal, List, Tuple, Union
from os import makedirs, path, listdir
from argparse import ArgumentParser, Namespace
import yaml
import cv2

# Constants for config file.
MAX_N = "all"

# Custom types.
Labels = Dict[str, List[int]]


def args_parser() -> Namespace:
    """Parse arguments.
        1. (-i) -> Input path
        2. (-o) -> Output path
        3. (-n) -> N of labels to get
        4. (-c) -> Dataset config

    Returns:
        Namespace: parsed arguments.
    """
    # Config the argparser and get the args.
    parser = ArgumentParser(description="Convert video to frames.")
    parser.add_argument("-i", "--input", help="Input path", required=False, type=str)
    parser.add_argument("-o", "--output", help="Output path", required=False, type=str)
    parser.add_argument("-n", "--labels", help="Number of images", required=False)
    parser.add_argument(
        "-c", "--config", help="Dataset config file (labels info)", required=False
    )
    args = parser.parse_args()
    return args


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


def raise_error(error: str) -> None:
    """Raises custom Video2Frame error.

    Args:
        error (str): specifi error to show.

    Raises:
        ValueError: Error of script.
    """
    raise ValueError(f" Video2Frame: {error}")


def log(
    msg: str,
    mode: Literal[
        "info",
        "warning",
    ] = "info",
    delete_previous: bool = False,
) -> None:
    """Log msg.

    Args:
        msg (str): msg to log.
        type (Literal[, optional): type of msg. Defaults to "Info".
        delete_previous (bool, optional): delete previous log. Defaults to False.
    """
    print(f"({mode}) - Video2Frame: {msg}", end="\r" if delete_previous else "\n")


def is_number(string: Union[str, int]) -> bool:
    """Check if string is number.

    Args:
        s (str): string to check.

    Returns:
        bool: string is number.
    """
    try:
        int(string)
        return True
    except ValueError:
        return False


def read_config() -> dict:
    """Read config file.

    Returns:
        dict: config file.
    """
    # Read config.yaml in case agruments are not recieved.
    config_file = "config.yml"
    yaml_file_path = abs_path(config_file)

    with open(yaml_file_path, encoding="utf8") as config_file:
        config = yaml.safe_load(config_file)
    return config


def check_path(
    custom_path: str,
    name: str,
) -> str:
    """Check if path exists.

    Args:
        custom_path (str): path to check.
        name (str): name of path.
    """
    custom_abs_path = abs_path(custom_path)
    if not path.exists(custom_abs_path):
        raise_error(
            f"Invalid {name} path ({custom_path}). Check is correct and exists."
        )
    return custom_abs_path


def get_videos(input_path: str) -> List[str]:
    """Get videos from input path.

    Args:
        input_path (str): path to get videos from.

    Returns:
        List[str]: videos path.
    """
    # The number of videos should be more than 0.
    videos = [vid for vid in listdir(input_path) if vid.endswith(".mp4")]
    n_videos = len(videos)
    if n_videos == 0:
        raise_error(f"No videos found in input path ({input_path}).")
    return videos


def check_number(number_labels: Union[str, int], max_n_labels: int) -> int:
    """Check if number of labels is valid.

    Args:
        number_labels (Union[str, int]): number of labels.
        max_n_labels (int): number of video labels.

    Returns:
        int: number of labels.
    """
    n_labels = 0

    # Check validity of number of images "all" or number.
    if is_number(number_labels) and int(number_labels) <= 0:
        raise_error(
            f"Invalid number of labels ({number_labels}). Cannot use negative values or 0."
        )
    elif not is_number(number_labels) and number_labels != MAX_N:
        raise_error(f'Invalid string ({number_labels}). Valid string is "{MAX_N}".')
    elif number_labels == MAX_N or int(number_labels) > max_n_labels:
        if number_labels == MAX_N:
            log(
                f'Detected "{MAX_N}" string for number of labels. Using max value ({max_n_labels}).',
            )
        else:
            log(
                f"Number of labels ({number_labels}) is larger than number of valid labels ({n_labels}). Using {max_n_labels}."
            )
        n_labels = max_n_labels
    else:
        n_labels = int(number_labels)

    return n_labels


def get_videos_path_and_name(
    input_path: str, output_path: str, labels: Labels, n_labels: int
) -> Tuple[List[str], List[str]]:
    """Get videos path and name.

    Args:
        input_path (str): path to get videos from.
        output_path (str): path to extract the frames to.
        labels (Labels): labels info.
        n_labels (int): number of labels.

    Returns:
        Tuple[List[str], List[str]]: videos input and output path.
    """

    input_paths = []
    output_paths = []
    for i, (label, ids) in enumerate(labels.items()):
        if i > n_labels:
            break
        for id in ids:
            input_paths.append(abs_path(path.join(input_path, f"{id}.mp4")))

            out_dir = path.join(output_path, label)
            if not path.exists(out_dir):
                makedirs(out_dir)

            output_paths.append(abs_path(path.join(out_dir, f"{id}.png")))

    return input_paths, output_paths


def extract_frames(
    videos_input_path: List[str],
    videos_output_path: List[str],
) -> None:
    """Extract frames from videos.

    Args:
        videos_input_path (List[str]): videos input path.
        videos_output_path (List[str]): videos output path.
    """
    bar_len = 20

    for i, (in_path, out_path) in enumerate(zip(videos_input_path, videos_output_path)):
        cap = cv2.VideoCapture(in_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = frame_count // 2

        perc = (i + 1) / len(videos_input_path) * 100
        normal_perc = perc * bar_len / 100
        perc_str = "{:.2f}".format(perc)
        log(
            f"Converting {i+1}/{len(videos_input_path)} videos - ({perc_str}%) [{'#' * int(normal_perc):<{bar_len}}]",
            delete_previous=True,
        )

        cap.set(1, mid_frame)
        (_, frame) = cap.read()
        cv2.imwrite(out_path, frame)


def load_labels(config_path: str, n_videos: int) -> Labels:
    """Load labels from config file.

    Args:
        config_path (str): path to config file.
        n_videos (int): number of videos that should have config info.

    Returns:
        Labels: labels from config file.
    """
    with open(config_path, encoding="utf8") as ipf:
        content = json.load(ipf)

    labels: Labels = {}
    n_videos_config = 0

    for entry in content:
        label = entry["gloss"]
        labels[label] = [video["video_id"] for video in entry["instances"]]
        n_videos_config += len(labels[label])

    if not n_videos_config == n_videos:
        raise_error(
            f"Number of videos in config file ({n_videos_config}) does not match number of videos ({n_videos}) in input folder."
        )

    return labels


def main(
    input_path: str, output_path: str, number_labels: Union[str, int], config_path: str
) -> None:
    """Main function.

    Args:
        input_path (str): path to input videos.
        output_path (str): path to output frames.
        number_labels (Union[str, int]): number of labels.
        config_path (str): path to config file.
    """
    if not input_path or not output_path or not number_labels or not config_path:
        raise_error(
            "Invalid values (empty or wrong types). Check config file or provide arguments via terminal (-h for help)."
        )

    input_path = check_path(input_path, "input")
    videos = get_videos(input_path)

    output_path = check_path(output_path, "output")

    # Load labels from config.
    config_path = check_path(config_path, "config")
    labels = load_labels(config_path, n_videos=len(videos))

    # Get correct number of labels.
    number_labels = check_number(number_labels, max_n_labels=len(labels))

    # Get videos path and name structures as a pytorch dataset directory.
    videos_input_path, videos_output_path = get_videos_path_and_name(
        input_path, output_path, labels, number_labels
    )

    extract_frames(videos_input_path, videos_output_path)


if __name__ == "__main__":
    args = args_parser()
    config = read_config()

    # If no args are provided, try to get from config.yaml.
    input_path = args.input or config["input"]
    output_path = args.output or config["output"]
    number_labels = args.labels or config["number_labels"]
    config_path = args.config or config["wlasl_config"]

    main(input_path, output_path, number_labels, config_path)

# TODO:
# - Improve checking on input, output and config paths (config is JSON, etc).
