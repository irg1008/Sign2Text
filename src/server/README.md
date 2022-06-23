# Server for delivery of onnx model to frontend

Made with fastapi as an alternative to a jsvascript-made backend using onnx.js

This is because onnx.js is not as relaible as using python based libraries.

the fonrt will fetch send a video and we will process it and send the target back.

No swagger or openapi standars are applied in this porject given we only have one endpoint for the inference.

## Run the backend server

`uvicorn main:app --reload`

## Tools used for testing

[Insomnia](https://insomnia.rest/download)

## Donde está hosteado el backend
