from fastapi import FastAPI, File, UploadFile, HTTPException
from utils.video import process_video, get_frames, create_tmp_path
from utils.onnx import get_session, oxx_inference
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI()

onnx_session = get_session()

# Origins for development and production clients.
origins = ["http://localhost:3000", "https://sign2text.com"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
    max_age=3600,
)


class TargetModel(BaseModel):
    """
    OpenAPI Model for target response.
    """

    target: str


@app.post(
    "/sign",
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
    frames = get_frames(file_tmp_path)

    # Process video for inference.
    frames = await process_video(frames)

    # Input to the model.
    target = oxx_inference(frames, onnx_session)

    return TargetModel(target=target)
