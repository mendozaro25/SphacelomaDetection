import cv2
import torch
from ultralytics import YOLO
import time
import os
import psutil
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment


class ObjectDetection:
    def __init__(self, model_path='yolov8n.pt', device='cpu', confidence_threshold=0.5, save_images=False):
        self.device = device
        self._validate_model_path(model_path)
        self.model = self._load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.save_images = save_images
        self.detections = []  # Lista para almacenar detecciones
        self.class_colors = self._generate_class_colors()

        if self.save_images:
            self.results_dir = self._create_results_dir()

        self._check_gpu()

    @staticmethod
    def _validate_model_path(model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"El modelo en {model_path} no existe.")

    def _load_model(self, model_path):
        model = YOLO(model_path)
        model.to(self.device)
        return model

    def _check_gpu(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            print("Usando CPU.")

    def _generate_class_colors(self):
        num_classes = len(self.model.model.names)
        return {i: tuple(map(int, torch.randint(0, 255, (3,)).tolist())) for i in range(num_classes)}

    def _create_results_dir(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = os.path.join('results', timestamp)
        os.makedirs(results_path, exist_ok=True)
        return results_path

    def _save_detection(self, frame):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(self.results_dir, f'detection_{timestamp}.jpg')
        cv2.imwrite(filename, frame)
        return filename

    def _save_excel(self):
        if not self.detections:
            print('No hay detecciones para guardar en Excel.')
            return

        filename = os.path.join(self.results_dir, 'detections.xlsx')
        wb = Workbook()
        ws = wb.active
        ws.title = "Detecciones"

        # Encabezados
        headers = ['Fecha y Hora', 'Archivo', 'Umbral']
        ws.append(headers)

        # Estilo de encabezados
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center')

        # Datos de detecciones
        for detection in self.detections:
            ws.append([detection[0], detection[1], float(detection[2])])  # Asegúrate de que es un valor numérico

        wb.save(filename)
        print(f'Archivo Excel guardado: {filename}')

    @staticmethod
    def _check_battery():
        battery = psutil.sensors_battery()
        return battery and battery.percent > 20

    @torch.no_grad()
    def detect_objects_from_camera(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError('No se pudo acceder al video o la cámara')

        prev_frame_time = time.time()
        while True:
            if not self._check_battery():
                print('La batería es baja. Cerrando la aplicación.')
                break

            ret, frame = cap.read()
            if not ret:
                print('Error al capturar el cuadro')
                break

            results = self.model.predict(source=frame, conf=self.confidence_threshold, device=self.device, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])  # Asegúrate de que es float
                    cls = int(box.cls[0])

                    color = self.class_colors.get(cls, (0, 255, 0))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f'{self.model.model.names[cls]}: {conf:.2f}'
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Guardar la imagen y la detección
                    if self.save_images and conf > self.confidence_threshold:
                        filename = self._save_detection(frame)
                        if filename:  # Solo si se guardó la imagen con éxito
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            self.detections.append([timestamp, filename, conf])

            # Calcular FPS
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Detecciones en tiempo real', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self._save_excel()
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    root = tk.Tk()
    root.withdraw()

    # Solicitar si se guardan imágenesq
    save_images = messagebox.askyesno("Guardar Imágenes", "¿Deseas guardar las imágenes con detecciones?")

    # Solicitar el umbral de confianza
    confidence_threshold = None
    while confidence_threshold is None:
        try:
            confidence_threshold = simpledialog.askfloat("Umbral de Confianza", "Ingresa el umbral de confianza (0.0 a 1.0):", minvalue=0.0, maxvalue=1.0)
        except ValueError:
            messagebox.showerror("Error", "Por favor ingresa un número válido para el umbral de confianza.")

    # Solicitar la fuente de video
    video_source = None
    while video_source is None:
        video_source_input = simpledialog.askstring("Fuente de Video", "Ingresa la fuente de video (0 para cámara, o la ruta del archivo):")
        if video_source_input.isdigit():
            video_source = int(video_source_input)
        else:
            messagebox.showerror("Error", "Por favor ingresa un número válido o la ruta del archivo de video.")

    detector = ObjectDetection(model_path='models/best.pt', confidence_threshold=confidence_threshold, save_images=save_images)
    detector.detect_objects_from_camera(video_source=video_source)


if __name__ == '__main__':
    main()
