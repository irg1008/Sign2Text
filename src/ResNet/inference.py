import torch
import torch.nn.functional as F
from PIL import Image
from config import device, transform
import cv2


def Webcam_720p(cap):
    cap.set(3, 1280)
    cap.set(4, 720)


def argmax(scores, classes):
    percentage = F.softmax(scores, dim=1)[0] * 100
    _, indices = torch.sort(scores, descending=True)
    first_five = [(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]

    # _, prediction = scores.max(1)
    # result = classes[prediction]

    # score = F.softmax(scores, dim=1)[0] * 100
    # score = score[prediction]

    return first_five


def preprocess(image):
    image = transform(image)
    image = image.float()
    image = image.to(device)
    image = torch.unsqueeze(image, 0)
    return image


def webcam_inference(model, classes):
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(0)  # Set the webcam
    Webcam_720p(cap)

    fps = 0
    show_score = 0
    first_five = []
    sequence = 0
    while True:
        ret, frame = cap.read()  # Capture each frame

        if fps == 15:
            image = frame  # [100:400, 150:550]
            image = Image.fromarray(image)
            image_data = preprocess(image)

            scores = model(image_data)
            first_five = argmax(scores, classes)

            fps = 0

        fps += 1

        y = 150
        for label, score in first_five:
            y += 50
            cv2.putText(
                frame,
                f"{label} - {score:.2f}",
                (900, y),
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


def path_inference(model, classes, img_path: str):
    image = Image.open(img_path).convert("RGB")
    image = preprocess(image)
    output = model(image)
    return argmax(output, classes)
