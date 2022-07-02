# UBU-Sign2Text

<table align="center"><tr><td align="center" width="9999">

<br />

<img align="center" src="./docs/assets/logo/logo.svg" alt="logo" width="400" />

<br />
<br />

[![Vercel](https://therealsujitk-vercel-badge.vercel.app/?app=sign2text-irgazquez&style=flat)](https://sign2text.com)
[![Api working](https://img.shields.io/badge/api-working-brightgreen?logo=fastapi)](https://api.sign2text.com/docs)
![Deployed on](https://img.shields.io/badge/model-google_cloud-brightgreen?logo=google-cloud)
[![Dockerized](https://img.shields.io/badge/docker-container-2496ED?logo=docker)](https://hub.docker.com/repository/docker/gazquez/sign2text)

![Pipeline Status](https://gitlab.com/HP-SCDS/Observatorio/2021-2022/sign2text/ubu-sign2text/badges/main/pipeline.svg)
[![License](https://img.shields.io/github/license/irg1008/sign2text)](https://gitlab.com/HP-SCDS/Observatorio/2021-2022/sign2text/ubu-sign2text/-/blob/main/LICENSE)
![Languages](https://img.shields.io/github/languages/count/irg1008/sign2text?logo=python)
![Top language](https://img.shields.io/github/languages/top/irg1008/sign2text?logo=jupyter)
![Lines of code](https://img.shields.io/badge/lines_of_code-3.7k-blueviolet)

![Front with astro](https://img.shields.io/badge/front_end-astro-orange?logo=astro)
![runtime onnx](https://img.shields.io/badge/runtime-onnx-lightgray?logo=onnx)

Transcripción de lenguaje de signos (a nivel de palabra) mediante Deep Learning

Check the [front-end](https://github.com/irg1008/Sign2Text-Astro) and [back-end](https://github.com/irg1008/Sign2Text-API) repos
</td></tr></table>

---

## Índice de contenido

- [UBU-Sign2Text](#ubu-sign2text)
  - [Índice de contenido](#índice-de-contenido)
  - [Antes de nada - Pruébalo](#antes-de-nada---pruébalo)
  - [Instalación](#instalación)
    - [Instalamos el paquete de entorno virtual](#instalamos-el-paquete-de-entorno-virtual)
    - [Creamos un entorno virtual](#creamos-un-entorno-virtual)
    - [Activamos el entorno virtual](#activamos-el-entorno-virtual)
    - [Instalamos los paquetes necesarios](#instalamos-los-paquetes-necesarios)
    - [Instalamos PyTorch compatible con CUDA (en caso de que queramos usar GPU)](#instalamos-pytorch-compatible-con-cuda-en-caso-de-que-queramos-usar-gpu)

---

## Antes de nada - Pruébalo

Puedes acceder a la API del modelo en [api.sign2text.com/docs](https://api.sign2text.com/docs).

También puedes subir un video en [sign2text.com](https://sign2text.com) para probarlo.
Ten en cuenta que la API del modelo consume muchos recursos (y tiene pocos asignados).
No va a poder procesar videos largos ni muchas peticiones simultáneas.

Si deseas más información sobre la API o sobre el front-end, puedes visitar los respositorios en:

- API: [Sign2Text API (Github)](https://github.com/irg1008/Sign2Text-API)
- FrontEnd: [Sign2Text FrontEnd (Github)](https://github.com/irg1008/Sign2Text-Astro)

## Instalación

Se usa un entorno virtual de Python para la ejecución de este programa.

Esto se puede hacer del siguiente modo:

### Instalamos el paquete de entorno virtual

```bash
pip install virtualenv
```

### Creamos un entorno virtual

```bash
virtualenv venv
```

### Activamos el entorno virtual

> Linux:

```bash
source venv/bin/activate
```

> Windows:

```bash
./venv/Scripts/activate
```

### Instalamos los paquetes necesarios

```bash
pip install -r requirements.txt
```

### Instalamos PyTorch compatible con CUDA (en caso de que queramos usar GPU)

Esto nos instalará los binarios de CUDA y CuDNN.

```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
