import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Cargar el clasificador Haar Cascade para detección facial
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar rostros
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces, gray
    
    def extract_face_embedding(self, face_roi):
        # En una implementación real, aquí usarías un modelo como FaceNet
        # Para este ejemplo, usaremos un embedding simple basado en histogramas
        face_resized = cv2.resize(face_roi, (100, 100))
        hist = cv2.calcHist([face_resized], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def draw_faces(self, frame, faces, recognized_name=None):
        for (x, y, w, h) in faces:
            # Dibujar rectángulo alrededor del rostro
            color = (0, 255, 0) if recognized_name else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            if recognized_name:
                # Mostrar nombre reconocido
                cv2.putText(frame, recognized_name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            else:
                cv2.putText(frame, 'Rostro Detectado', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return frame