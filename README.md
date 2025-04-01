# Object Detection with YOLOv8

Este proyecto implementa detección de objetos en tiempo real utilizando YOLOv8 y OpenCV. Permite capturar video desde una cámara o archivo, procesarlo con un modelo preentrenado y guardar imágenes y resultados en un archivo Excel.

## Requisitos

- Python 3.8+
- pip
- Dependencias necesarias:
  ```bash
  pip install -r requirements.txt
  ```
  
## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/mendozaro25/SphacelomaDetection.git
   cd SphacelomaDetection
   ```
2. Instala los paquetes requeridos:
   ```bash
   pip install -r requirements.txt
   ```
3. Descarga el modelo YOLOv8 y colócalo en `models/best.pt`.

## Uso

Ejecuta el script principal:
```bash
python main.py
```
El sistema te preguntará si deseas guardar imágenes, el umbral de confianza y la fuente de video.

- Para salir de la detección en tiempo real, presiona `q`.
- Los resultados se guardan en la carpeta `results/`.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.


