from typing import List
from onnx import load, ModelProto
from os import path
from onnxruntime import InferenceSession


CLASSES = ["all", "before", "book", "deaf", "drink", "help", "no", "walk", "yes"]
MODEL_PATH = path.join(path.dirname(__file__), "../../models/WLASL_9_quantized.onnx")

ONNX_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def get_model() -> ModelProto:
    """Get the model from the onnx file.

    Returns:
        ModelProto: ONNX Model.
    """
    model = load(MODEL_PATH)
    return model


def get_session() -> InferenceSession:
    """Get the session from the onnx file.

    Returns:
        InferenceSession: Session ready for inference.
    """
    ort_session = InferenceSession(
        MODEL_PATH,
        providers=ONNX_PROVIDERS,
    )
    return ort_session


def oxx_inference(video: List, session: InferenceSession) -> str:
    """Inference the video with the onnx model.

    Args:
        video (List): Video to inference.
        session (InferenceSession): Session ready for inference.

    Returns:
        str: Class of the video.
    """
    outputs = session.run(
        None,
        {"input": video},
    )
    return CLASSES[outputs[0][0].argmax(0)]
