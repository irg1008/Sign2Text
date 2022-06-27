from typing import List
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


def webcam_720p(cap):
    """Set the webcam to 720p.

    Args:
        cap (VideoCapture): VideoCapture object.
    """
    cap.set(3, 1280)
    cap.set(4, 720)


def argmax(scores, classes):
    """Get the labels with the highest score.

    Args:
        scores (Tensor): Scores of the model.
        classes (List[str]): List of target classes.

    Returns:
        List[str]: First five labels of the image.
    """
    percentage = F.softmax(scores, dim=1)[0] * 100
    _, indices = torch.sort(scores, descending=True)
    first_five = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    return first_five


def preprocess(image, device, transform):
    """Preprocess a video or image given the pytorch transform function.

    Args:
        image (Tensor): Image or video to transform.
        device ("cpu" | "gpu"): Device to transform the tensor in.
        transform (Transform): Pytorch transform function.

    Returns:
        Tensor: transformed input tensor.
    """
    image = transform(image)
    image = image.float()
    image = image.to(device)
    image = torch.unsqueeze(image, 0)
    return image


def video_webcam_inference(
    model, classes, device, transform, fps_interval: int, has_pose=False
):
    """Infere a video from webcam.

    Args:
        model (Model): Video model to infere with
        classes (List[str]): List of target classes.
    """
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)  # Set the webcam
    webcam_720p(cap)

    IMAGE_SIZE = 600

    upper_left = (300, 50)
    bottom_right = (upper_left[0] + IMAGE_SIZE, upper_left[1] + IMAGE_SIZE)

    first_five = []
    video: List = []
    fps = 0
    while True:
        _, frame = cap.read()  # Capture each frame
        # Cut frame.
        rect_frame = frame[
            upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]
        ]

        fps += 1

        # Save all frames every 30 frames and feed the net.
        video.append(rect_frame)

        # Reset every 'fps_interval' frames.
        if fps % fps_interval == 0:
            transformed_video = preprocess(video, device, transform)
            scores = (
                model(transformed_video)[0] if has_pose else model(transformed_video)
            )

            first_five = argmax(scores, classes)
            video = []

        screen_y = 150
        for label, score in first_five:
            screen_y += 50
            cv2.putText(
                frame,
                f"{label} - {score:.2f}",
                (950, screen_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.rectangle(frame, upper_left, bottom_right, (250, 0, 0), 2)
        cv2.imshow("ASL SIGN DETECTER", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyWindow("ASL SIGN DETECTER")


def webcam_inference(model, classes, device, transform):
    """Infere an image from webcam.

    Args:
        model (Model): Image model to infere with
        classes (List[str]): List of target classes.
    """
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)  # Set the webcam
    webcam_720p(cap)

    fps = 0
    first_five = []
    while True:
        _, frame = cap.read()  # Capture each frame

        if fps == 15:
            image = frame  # [100:400, 150:550]
            image = Image.fromarray(image)
            image_data = preprocess(image, device, transform)

            scores = model(image_data)
            first_five = argmax(scores, classes)

            fps = 0

        fps += 1

        screen_y = 150
        for label, score in first_five:
            screen_y += 50
            cv2.putText(
                frame,
                f"{label} - {score:.2f}",
                (900, screen_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        # cv2.rectangle(frame, (400, 150), (900, 550), (250, 0, 0), 2)
        cv2.imshow("ASL SIGN DETECTER", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyWindow("ASL SIGN DETECTER")


def path_inference(model, classes, device, transform, img_path: str):
    """Inference an image given the image path.

    Args:
        model (Model): Image model to infere with
        classes (List[str]): List of target classes.
        img_path (str): Path of image.

    Returns:
        str: Label of the image.
    """
    image = Image.open(img_path).convert("RGB")
    image = preprocess(image, device, transform)
    output = model(image)
    return argmax(output, classes)
