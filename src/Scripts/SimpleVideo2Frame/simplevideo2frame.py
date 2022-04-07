from argparse import ArgumentParser, Namespace
from os import listdir, makedirs, path, popen
from typing import List, Tuple
import cv2
from common.utils.file import abs_path
from common.utils.log import log

Paths = List[Tuple[str, str]]
LOG_FILE = "./errors.log"


def args_parser() -> Namespace:
    """Argument parser.

    Returns:
        Namespace: parsed arguments.
    """
    parser = ArgumentParser(description="Convert video to frames.")
    parser.add_argument("-i", "--input", help="Input path", required=True, type=str)
    parser.add_argument("-o", "--output", help="Output path", required=True, type=str)
    parser.add_argument(
        "-l", "--labels", help="Number of labels to export", required=True
    )
    parser.add_argument(
        "-c",
        "--convert",
        help="Convert video to mp4",
        required=False,
        action="store_true",
    )
    return parser.parse_args()


def clean_log():
    """Clean log file."""
    log_file = open(LOG_FILE, "w", encoding="utf-8")
    log_file.truncate(0)
    log_file.close()


def write_to_log(msg: str):
    """Write msg to log file.

    Args:
        msg (str): msg to log.
    """
    log_file = open(LOG_FILE, "w", encoding="utf-8")
    log_file.write(msg + "\n")
    log_file.close()


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


def create_if_not_exists(path_to_create: str) -> bool:
    """Create path if not exists.

    Args:
        path_to_create (str): path to create.

    Returns:
        bool: True if created, False otherwise.
    """
    exists = path.exists(path_to_create)
    if not exists:
        makedirs(path_to_create)
    return not exists


def extract_video_frames(out_path: str, video_path: str):
    """Extract frames from video.

    Args:
        out_path (str): output path.
        video_path (str): video path.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    already_extracted = len(listdir(out_path))
    for frame in range(already_extracted, frame_count):
        cap.set(1, frame)
        ret, img = cap.read()
        img_path = path.join(out_path, f"img_{frame + 1:05d}.png")
        if ret:
            cv2.imwrite(img_path, img)
        else:
            write_to_log(
                f"Error extracting frame {frame} from {video_path} onto {img_path}"
            )


def export_videos(paths: Paths):
    """Export videos.

    Args:
        paths (Paths): paths to export.
    """
    for i, path in enumerate(paths):
        perc = (i + 1) / len(paths) * 100
        log(
            f"Converting {i+1}/{len(paths)} videos - ({perc:.2f}%) {get_loadbar(perc)}",
            delete_previous=True,
        )
        extract_video_frames(*path)


def convert2mp4(input_path: str, output_path: str, n_labels: int):
    """Convert videos to mp4.

    Args:
        input_path (str): input path.
        output_path (str): output path.
        n_labels (int): number of labels.
    """
    create_if_not_exists(output_path)

    for i, label in enumerate(listdir(input_path)):
        if i >= n_labels:
            break
        label_path = path.join(input_path, label)
        output_path = path.join(output_path, label)
        create_if_not_exists(output_path)

        for i, video in enumerate(listdir(label_path)):
            avi_video_path = path.join(label_path, video)
            mp4_video_path = path.join(output_path, video.replace(".avi", ".mp4"))
            # popen(f"ffmpeg -i {avi_video_path} -c:v libx264 -c:a copy {mp4_video_path}")
            popen(
                f"ffmpeg -i {avi_video_path} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {mp4_video_path}"
            )


def get_paths(input_path: str, output_path: str, n_labels: int) -> Paths:
    """Get paths.

    Args:
        input_path (str): input path.
        output_path (str): output path.
        n_labels (int): number of labels.

    Returns:
        Paths: paths.
    """
    create_if_not_exists(output_path)
    paths: Paths = []

    for i, label in enumerate(listdir(input_path)):
        if i >= n_labels:
            break
        label_path = path.join(input_path, label)
        create_if_not_exists(label_path)

        for j, video in enumerate(listdir(label_path)):
            out_path = path.join(output_path, label, f"{j + 1:04d}")
            video_path = path.join(label_path, video)

            create_if_not_exists(out_path)
            paths.append((out_path, video_path))

    return paths


def main(input_path: str, output_path: str, n_labels: int, convert_videos: bool):
    """Main function.

    Args:
        input_path (str): input path.
        output_path (str): output path.
        n_labels (int): number of labels.
        convert_videos (bool): convert videos flag.
    """
    input_path = abs_path(input_path)
    output_path = abs_path(output_path)

    if convert_videos:
        convert2mp4(input_path, output_path, n_labels)
    else:
        paths = get_paths(input_path, output_path, n_labels)
        export_videos(paths)


if __name__ == "__main__":
    clean_log()

    args = args_parser()

    input_path = args.input
    output_path = args.output
    number_labels = int(args.labels)
    convert_videos = args.convert

    main(input_path, output_path, number_labels, convert_videos)
