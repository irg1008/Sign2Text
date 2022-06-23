from typing import List
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


def webcam_720p(cap):
    """_summary_

    Args:
        cap (_type_): _description_
    """
    cap.set(3, 1280)
    cap.set(4, 720)


def argmax(scores, classes):
    """_summary_

    Args:
        scores (_type_): _description_
        classes (_type_): _description_

    Returns:
        _type_: _description_
    """
    percentage = F.softmax(scores, dim=1)[0] * 100
    _, indices = torch.sort(scores, descending=True)
    first_five = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    return first_five


def preprocess(image, device, transform):
    """_summary_

    Args:
        image (_type_): _description_

    Returns:
        _type_: _description_
    """
    image = transform(image)
    image = image.float()
    image = image.to(device)
    image = torch.unsqueeze(image, 0)
    return image


def video_webcam_inference(model, classes, device, transform, fps_interval: int):
    """_summary_

    Args:
        model (_type_): _description_
        classes (_type_): _description_
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
            scores, poses = model(transformed_video)

            # Multiply poses by image width.
            # POSES_PER_FRAME = 84  # See train notebook.
            # last_frame_poses = poses.squeeze()[-POSES_PER_FRAME:]
            # last_frame_poses *= IMAGE_SIZE
            # last_frame_poses = last_frame_poses.cpu().detach().numpy()
            # x, y = last_frame_poses[0::2], last_frame_poses[1::2]

            # Plot the poses.
            # plt.scatter(x, y)
            # plt.pause(0.001)

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

        # plt.imshow(rect_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyWindow("ASL SIGN DETECTER")


def webcam_inference(model, classes, device, transform):
    """_summary_

    Args:
        model (_type_): _description_
        classes (_type_): _description_
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
    """_summary_

    Args:
        model (_type_): _description_
        classes (_type_): _description_
        img_path (str): _description_

    Returns:
        _type_: _description_
    """
    image = Image.open(img_path).convert("RGB")
    image = preprocess(image, device, transform)
    output = model(image)
    return argmax(output, classes)
