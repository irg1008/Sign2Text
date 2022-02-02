from ast import List
from typing import Literal, List, Tuple, Union
import yaml
from os import path, listdir
from argparse import ArgumentParser, Namespace
import cv2


def args_parser() -> Namespace:
    """Parse arguments.
        1. (-i) -> Input path
        2. (-o) -> Output path
        3. (-n) -> N images

    Returns:
        Namespace: parsed arguments.
    """
    # Config the argparser and get the args.
    parser = ArgumentParser(description="Convert video to frames.")
    parser.add_argument("-i", "--input", help="Input path", required=False, type=str)
    parser.add_argument("-o", "--output", help="Output path", required=False, type=str)
    parser.add_argument("-n", "--images", help="Number of images", required=False)
    args = parser.parse_args()
    return args


def abs_path(custom_path: str) -> str:
    """Returns the absolute path of a path

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
    return path.abspath(path.join(path.abspath(path.dirname(__file__)), custom_path))


def raise_error(error: str) -> None:
    """Raises custom Video2Frame error.

    Args:
        error (str): specifi error to show.

    Raises:
        ValueError: Error of script.
    """
    raise ValueError(f" Video2Frame: {error}")


def log(msg: str, type: Literal["Info", "Warning"] = "Info") -> None:
    """Log msg.

    Args:
        msg (str): msg to log.
        type (Literal[, optional): type of msg. Defaults to "Info".
    """
    print(f"({type}) - Video2Frame: {msg}")


def is_number(s: Union[str, int]) -> bool:
    """Check if string is number.

    Args:
        s (str): string to check.

    Returns:
        bool: string is number.
    """
    try:
        int(s)
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
    config = yaml.safe_load(open(yaml_file_path))
    return config


def check_path(custom_path: str, name: str) -> str:
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


def check_number(
    number_images: Union[str, int], input_path: str
) -> Tuple[List[str], int]:
    """Check number is correct.

    Args:
        number_images (Union[str, int]): number of videos wanted. Should be a number or the constant value "all".
        input_path (str): path the videos are in.

    Returns:
        List[str]: n videos for input path.
    """
    MAX_N = "all"
    n_images = 0

    # The number of videos should be more than 0.
    videos = [vid for vid in listdir(input_path) if vid.endswith(".mp4")]
    n_videos = len(videos)
    if n_videos == 0:
        raise_error(f"No videos found in input path ({input_path}).")

    # Check validity of number of images "all" or number.
    if is_number(number_images) and int(number_images) <= 0:
        raise_error(
            f"Invalid number of images ({number_images}). Cannot use negative values or 0."
        )
    elif not is_number(number_images) and number_images != MAX_N:
        raise_error(f'Invalid string ({number_images}). Valid string is "{MAX_N}".')
    elif number_images == MAX_N or int(number_images) > n_videos:
        if number_images == MAX_N:
            log(
                f'Detected "{MAX_N}" string for number of images. Using max value ({n_videos}).'
            )
        else:
            log(
                f"Number of images ({number_images}) is larger than number of videos in folder ({n_videos}). Using {n_videos}."
            )
        n_images = n_videos
    else:
        n_images = int(number_images)

    return videos, n_images


def get_videos_path_and_name(
    input_path, videos, n_videos
) -> Tuple[List[str], List[str]]:
    """Get videos path and name.

    Args:
        input_path ([type]): path to get videos from.
        videos ([type]): list of videos.
        n_videos ([type]): number of videos.

    Returns:
        Tuple[List[str], List[str]]: videos path and name.
    """
    # Videos absolute path.
    paths = [abs_path(path.join(input_path, vid)) for vid in videos][:n_videos]

    # Video names.
    names = [vid.split(".")[0] for vid in videos][:n_videos]

    return paths, names


def extract_frames(videos_path, videos_name, output_path) -> None:
    """Extract frames from videos.

    Args:
        videos_path (List[str]): list of videos path.
        videos_name (List[str]): list of videos name.
        output_path (str): path to output frames.
    """
    for vid_path, vid_name in zip(videos_path, videos_name):
        cap = cv2.VideoCapture(vid_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = frame_count // 2

        cap.set(1, mid_frame)
        _, frame = cap.read()
        cv2.imwrite(f"{output_path}/{vid_name}.png", frame)


if __name__ == "__main__":
    args = args_parser()
    config = read_config()

    # If no args are provided, try to get from config.yaml.
    input_path = args.input or config["input"]
    output_path = args.output or config["output"]
    number_images = args.images or config["number_images"]

    # If any of previous values is not provided, raise error.
    if not input_path or not output_path or not number_images:
        raise_error(
            "Invalid values (empty or wrong types). Check config file or provide arguments via terminal (-h for help)."
        )

    input_path = check_path(input_path, "input")
    output_path = check_path(output_path, "output")
    videos, number_images = check_number(number_images, input_path)

    videos_path, videos_name = get_videos_path_and_name(
        input_path, videos, number_images
    )

    extract_frames(videos_path, videos_name, output_path)
