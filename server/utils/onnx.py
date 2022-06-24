from os import path
from onnx import load, ModelProto
from onnxruntime import InferenceSession
from torch import Tensor

CLASSES = ["all", "before", "book", "drink", "help", "no", "walk", "yes"]
MODEL_PATH = path.join(path.dirname(__file__), "../models/WLASL_8_quantized.onnx")

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


def oxx_inference(video: Tensor, session: InferenceSession) -> str:
    """Inference the video with the onnx model.

    Args:
        video (List): Video to inference.
        session (InferenceSession): Session ready for inference.

    Returns:
        str: Class of the video.
    """
    outputs = session.run(
        None,
        {"input": video.numpy()},
    )
    return CLASSES[outputs[0][0].argmax(0)]
