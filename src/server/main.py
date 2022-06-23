from fastapi import FastAPI, File, UploadFile, HTTPException
from utils.video import process_video, get_frames, create_tmp_path
from utils.onnx import get_session, oxx_inference
from pydantic import BaseModel

app = FastAPI()

onnx_session = get_session()


class TargetModel(BaseModel):
    target: str


@app.post(
    "/",
    tags=["video"],
    description="Video classification",
    response_model=TargetModel,
)
async def inference(video: UploadFile = File()) -> TargetModel:
    """Video classification.

    Args:
        video (UploadFile, optional): Video to inference.

    Raises:
        HTTPException: If the video is not a video.

    Returns:
        TargetModel: Class of the video.
    """
    if not video.content_type in ["video/mp4"]:
        raise HTTPException(status_code=400, detail="File must be mp4 video")

    # Get video frames.
    file_tmp_path = await create_tmp_path(video)
    frames, num_frames = get_frames(file_tmp_path)

    if num_frames < 50:
        raise HTTPException(
            status_code=400,
            detail="Video must have at least 50 frames. This is for ONNX inference. Shorter videos will work with PyTorch inference.",
        )

    # Process video for inference.
    frames = await process_video(frames)

    # Input to the model.
    target = oxx_inference(frames, onnx_session)

    return TargetModel(target=target)
