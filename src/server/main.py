from typing import Union
from fastapi import FastAPI, File, UploadFile, HTTPException


app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/inference")
async def inference(file: UploadFile = File()):
    print(file.content_type)
    # check file is video.
    if not file.content_type in ["video/mp4"]:
        raise HTTPException(status_code=400, detail="File must be mp4 video")
    return {"Hello": "World"}
