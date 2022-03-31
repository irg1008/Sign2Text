# VIDEO2FRAME

Este script selecciona el frame de un video de un conjunto de videos y lo exporta a una carpeta.
La carpeta de entrada y salida se puede pasar por parámetro o se puede especificar en el archivo de configuración.

---

## Ejecución

Podemos ejecutar el script dle siguiente modo:

```bash
  python video2frame.py
```

## Entrada por archivo de configuración

Al ejecutar de este modo se seleccionaran los datos del archivo `config.yaml`

## Entrada por parámetros

Si queremos ejecutar el script con parámetros, podemos pasarle los siguientes parámetros:

GLOBALES

- -i, --input: Carpeta con los videos a procesar (Opcional).
- -o, --output: Carpeta donde se guardaran los frames (Opcional).
- -l, --labels: Número de labels a procesar. Puede ser un número o "all" para procesar todos (Opcional).
- -c, --config: Archivo de configuración con las etiquetas (Opcional).
- -h, --help: Muestra esta ayuda.

EXTRACCIÓN DE FRAMES

- -f, --frames: Número de frames aleatorios del video (Opcional).
- -m, --merge: Indica si se debe unir los frames en una imagen (Opcional).

EXTRACCIÓN DE VIDEOS

- -v, --video: Extrae todos los frames del video en una carpeta única para ese video. Deshabilita las opciones -f y -m (Opcional).

### Ejemplos

```bash
python video2frame.py -i ./videos_input/ -o ./frames_output/ -l all
```

```bash
python video2frame.py -l 10 -c ./config.json
```

```bash
python video2frame.py -o ./frames_output/
```

```bash
python video2frame.py -l 40
```

```bash
python video2frame.py -f 3 --merge
```

```bash
python video2frame.py -o ./videos_output/ --videos -l 50
```
