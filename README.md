# UBU-Sign2Text

![Pipeline Status](https://gitlab.com/HP-SCDS/Observatorio/2021-2022/sign2text/ubu-sign2text/badges/main/pipeline.svg)

Transcripción de lenguaje de signos (a nivel de palabra) mediante Deep Learning

## Índice de contenido

* [Instalación](#instalación)
  * [Instalamos el paquete de entorno virtual](#instalamos-el-paquete-de-entorno-virtual)
  * [Creamos el entorno virtual](#creamos-el-entorno-virtual)
  * [Activamos el entorno virtual](#activamos-el-entorno-virtual)
  * [Instalamos los paquetes necesarios](#instalamos-los-paquetes-necesarios)
  * [Instalamos PyTorch compatible con CUDA](#instalamos-pytorch-compatible-con-cuda)

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

### Instalamos PyTorch compatible con CUDA

Esto nos instalará los binarios de CUDA y CuDNN.

```bash
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio===0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
