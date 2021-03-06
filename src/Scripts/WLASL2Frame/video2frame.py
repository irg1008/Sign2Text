from typing import Dict, List, Tuple, Union
from os import makedirs, path, listdir
from argparse import ArgumentParser, Namespace
import sys
import json
import shutil
import yaml
import cv2
import numpy as np

sys.path.append("../")

from common.utils.file import abs_path
from common.utils.log import log


# Constants for config file.
MAX_N = "all"

# Custom types.
Labels = Dict[str, List[int]]


def args_parser() -> Namespace:
    """Parse arguments.
        1. (-i) -> Input path
        2. (-o) -> Output path
        3. (-p) -> Poses path
        4. (-l) -> N of labels to get
        5. (-c) -> Dataset config
        6. (-f) -> Number of frames to extract
        7. (-m) -> Merge frames into one single image
        8. (-v) -> Export videos instead of images (Disables frame extraction)

    Returns:
        Namespace: parsed arguments.
    """
    # Config the argparser and get the args.
    parser = ArgumentParser(description="Convert video to frames.")
    parser.add_argument("-i", "--input", help="Input path", required=False, type=str)
    parser.add_argument("-o", "--output", help="Output path", required=False, type=str)
    parser.add_argument("-p", "--poses", help="Poses path", required=False, type=str)
    parser.add_argument("-l", "--labels", help="Number of images", required=False)
    parser.add_argument(
        "-c", "--config", help="Dataset config file (labels info)", required=False
    )
    parser.add_argument(
        "-f", "--frames", help="Number of frames to extract", required=False
    )
    parser.add_argument(
        "-m",
        "--merge",
        help="Merge frames into one single image",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-v",
        "--videos",
        help="Export videos instead of images, creating a folder for each video",
        required=False,
        action="store_true",
    )
    args = parser.parse_args()
    return args


def raise_error(error: str) -> None:
    """Raises custom Video2Frame error.

    Args:
        error (str): specifi error to show.

    Raises:
        ValueError: Error of script.
    """
    raise ValueError(f" Video2Frame: {error}")


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
    config_file_path = "config.yml"
    yaml_file_path = abs_path(path.join("./", config_file_path))

    config_file = open(yaml_file_path, "r", encoding="utf-8")
    config = yaml.safe_load(config_file)
    config_file.close()

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
    custom_abs_path = abs_path(path.join("./", custom_path))
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


def check_number(
    number_labels: Union[str, int], max_n_labels: int, use_log: bool = True
) -> int:
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
        if use_log:
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


def get_poses_path(pose_path: str, labels: Labels) -> List[str]:
    """Get poses from poses path.

    Args:
        pose_path (str): path to get poses from.
        poses (List[str]): poses path.

    Returns:
        List[str]: poses path.
    """
    input_paths = []

    for (_, ids) in labels.items():
        for vid_id in ids:
            input_paths.append(abs_path(path.join(pose_path, f"{vid_id}")))

    return input_paths


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
        if i >= n_labels:
            break
        for vid_id in ids:
            input_paths.append(abs_path(path.join(input_path, f"{vid_id}.mp4")))

            out_dir = path.join(output_path, label)
            if not path.exists(out_dir):
                makedirs(out_dir)

            output_paths.append(abs_path(out_dir))

    return input_paths, output_paths


def get_loadbar(perc: float) -> str:
    """Get loadbar.

    Args:
        perc (float): percentage.

    Returns:
        str: loadbar.
    """
    bar_len = 20
    normal_perc = perc * bar_len / 100
    return f"[{'#' * int(normal_perc):<{bar_len}}]"


def extract_pose_frames(in_pos: str, out_pos: str) -> None:
    """Extract pose frames.

    Args:
        in_pos (str): input pose path.
        out_pos (str): output pose path.
    """
    for i, pos in enumerate(listdir(in_pos)):
        new_name = f"img_{i + 1:05d}.json"
        shutil.copy(path.join(in_pos, pos), path.join(out_pos, new_name))


def extract_frames(
    in_path: str, out_path: str, number_frames: Union[str, int], merge: bool
):
    """Extract frames from video.

    Args:
        in_path (str): input path.
        out_path (str): output path.
        number_frames (int): number of frames.
        merge (bool): merge
    """

    def get_frames(frame_count: int) -> List[int]:
        n_frames = check_number(number_frames, max_n_labels=frame_count, use_log=False)
        padding = int(frame_count * 0.2)
        frames = (
            np.linspace(padding, frame_count - padding, n_frames, dtype=int).tolist()
            if n_frames > 1
            else [frame_count // 2]
        )
        return frames

    cap = cv2.VideoCapture(in_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    images = []
    for frame in get_frames(frame_count):
        cap.set(1, frame)
        _, img = cap.read()
        images.append(img)

    n_files_out_path = len(listdir(out_path)) + 1

    if merge:
        merged_img = np.concatenate(images, axis=1)
        cv2.imwrite(path.join(out_path, f"img_{n_files_out_path:05d}.png"), merged_img)
    else:
        for i, img in enumerate(images):
            cv2.imwrite(path.join(out_path, f"img_{n_files_out_path + i:05d}.png"), img)


def extract_video_frames(
    videos_input_path: List[str],
    videos_output_path: List[str],
    number_frames: int,
    merge: bool,
) -> None:
    """Extract frames from videos.

    Args:
        videos_input_path (List[str]): videos input path.
        videos_output_path (List[str]): videos output path.
    """

    for i, (in_path, out_path) in enumerate(zip(videos_input_path, videos_output_path)):
        perc = (i + 1) / len(videos_input_path) * 100
        log(
            f"Converting {i+1}/{len(videos_input_path)} videos - ({perc:.2f}%) {get_loadbar(perc)}",
            delete_previous=True,
        )
        extract_frames(in_path, out_path, number_frames, merge)

    log(
        f"All {len(videos_input_path)} videos converted successfully",
        double_new_line=True,
    )


def load_labels(config_path: str, n_videos: int) -> Labels:
    """Load labels from config file.

    Args:
        config_path (str): path to config file.
        n_videos (int): number of videos that should have config info.

    Returns:
        Labels: labels from config file.
    """
    ipf = open(config_path, "r", encoding="utf-8")
    content = json.load(ipf)
    ipf.close()

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


def copy_videos(
    in_path: List[str], out_path: List[str], poses_paths: List[str]
) -> None:
    """Copy videos from input path to output path.

    Args:
        in_path (List[str]): videos input path.
        out_path (List[str]): videos output path.
    """
    for i, (in_vid, out_vid, in_pos) in enumerate(zip(in_path, out_path, poses_paths)):
        n_files_out_path = len(listdir(out_vid)) + 1

        out_vid = path.join(out_vid, f"{n_files_out_path:04d}")
        if not path.exists(out_vid):
            makedirs(out_vid)

        extract_frames(in_vid, out_vid, "all", merge=False)
        extract_pose_frames(in_pos, out_vid)

        perc = (i + 1) / len(in_path) * 100
        log(
            f"Copying {i+1}/{len(in_path)} videos - ({perc:.2f}%) {get_loadbar(perc)}",
            delete_previous=True,
        )


def main(
    input_path: str,
    output_path: str,
    poses_path: str,
    number_labels: Union[str, int],
    config_path: str,
    number_frames: int,
    merge: bool,
    export_videos: bool,
) -> None:
    """Main function.

    Args:
        input_path (str): path to input videos.
        output_path (str): path to output frames.
        number_labels (Union[str, int]): number of labels.
        config_path (str): path to config file.
    """
    if (
        not input_path
        or not output_path
        or not poses_path
        or not number_labels
        or not config_path
    ):
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

    if export_videos:
        poses_path = check_path(poses_path, "poses")
        poses_paths = get_poses_path(poses_path, labels)
        copy_videos(videos_input_path, videos_output_path, poses_paths)
    else:
        extract_video_frames(
            videos_input_path, videos_output_path, number_frames, merge
        )


if __name__ == "__main__":
    args = args_parser()
    config = read_config()

    # If no args are provided, try to get from config.yaml.
    input_path = args.input or config["input"]
    poses_path = args.poses or config["poses"]
    output_path = args.output or config["output"]
    number_labels = args.labels or config["number_labels"]
    config_path = args.config or config["wlasl_config"]
    number_frames = args.frames or config["number_frames"]
    merge_frames = args.merge or config["merge_frames"]
    export_videos = args.videos or config["export_videos"]

    main(
        input_path,
        output_path,
        poses_path,
        number_labels,
        config_path,
        number_frames,
        merge_frames,
        export_videos,
    )
