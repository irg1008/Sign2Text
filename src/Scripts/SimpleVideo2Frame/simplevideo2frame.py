from argparse import ArgumentParser, Namespace
from os import listdir, makedirs, path, popen
from typing import List, Literal, Tuple
import cv2

Paths = List[Tuple[str, str]]
LOG_FILE = "./errors.log"


def args_parser() -> Namespace:
    # Config the argparser and get the args.
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
    args = parser.parse_args()
    return args


def clean_log():
    log_file = open(LOG_FILE, "w")
    log_file.truncate(0)
    log_file.close()


def write_to_log(msg: str):
    log_file = open(LOG_FILE, "w")
    log_file.write(msg + "\n")
    log_file.close()


def log(
    msg: str,
    mode: Literal[
        "info",
        "warning",
    ] = "info",
    delete_previous: bool = False,
    double_new_line: bool = False,
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


def create_if_not_exists(path_to_create: str):
    exists = path.exists(path_to_create)
    if not exists:
        makedirs(path_to_create)
    return exists


def extract_video_frames(out_path: str, video_path: str):
    # Read video and extract all frames to out_path.
    # > With incremental names
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
    for i, path in enumerate(paths):
        perc = (i + 1) / len(paths) * 100
        log(
            f"Converting {i+1}/{len(paths)} videos - ({perc:.2f}%) {get_loadbar(perc)}",
            delete_previous=True,
        )
        extract_video_frames(*path)


def convert2mp4(input_path: str, output_path: str, n_labels: int):
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


def get_paths(input_path: str, output_path: str, n_labels: int):
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
