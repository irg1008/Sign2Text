# Server for delivery of onnx model to frontend

Made with fastapi as an alternative to a jsvascript-made backend using onnx.js
This is because onnx.js is not as reliable as using python based libraries.
The client will send a video and the server will process it and send the target back.

---

## Run the backend server

`uvicorn main:app --reload`

or

`python run.py`

## Tools used for testing

[Insomnia](https://insomnia.rest/download)

## Where is it hosted?

This server is hosted

# We need to move this outside of this repo, to a single location with the model and single requiremente.txt file and lall that
