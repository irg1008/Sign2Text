# Server for delivery of onnx model to frontend

Made with fastapi as an alternative to a jsvascript-made backend using onnx.js

This is because onnx.js is not as relaible as using python based libraries.

the fonrt will fetch send a video and we will process it and send the target back.

No swagger or openapi standars are applied in this porject given we only have one endpoint for the inference.

## Run the backend server

`uvicorn main:app --reload`

## Tools used for testing

[Insomnia](https://insomnia.rest/download)

TODO: Realtime inference. It's quick enough.
Do after frontend
296 ms, y lo haría cada 50 frames a 30fps
Problema: Sale siempre "deaf". Igual deberíamso quitar deaf de las clases. Aunque creo que es lo de pillar los 50 primeros frames en vez de 50 intercalados por el video.
