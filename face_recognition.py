import cv2
import numpy as np
from database import Database
from face_detection import FaceDetector
import pickle

class FaceRecognizer:
    def __init__(self):
        self.detector = FaceDetector()
        self.db = Database()
        self.known_faces = []
        self.known_names = []
        self.load_known_faces()
    
    def load_known_faces(self):
        """Cargar rostros conocidos desde la base de datos"""
        faces_data = self.db.get_all_faces()
        self.known_faces = []
        self.known_names = []
        
        for face_id, name, face_blob in faces_data:
            try:
                # Deserializar los datos de la cara
                face_data = pickle.loads(face_blob)
                self.known_faces.append(face_data)
                self.known_names.append(name)
            except:
                print(f"Error cargando datos de {name}")
    
    def recognize_face(self, frame):
        faces, gray = self.detector.detect_faces(frame)
        recognized_names = []
        
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            
            try:
                # Extraer características del rostro
                current_embedding = self.detector.extract_face_embedding(face_roi)
                
                # Comparar con rostros conocidos
                best_match = None
                best_distance = float('inf')
                
                for i, known_embedding in enumerate(self.known_faces):
                    # Calcular distancia euclidiana
                    distance = np.linalg.norm(current_embedding - known_embedding)
                    
                    if distance < best_distance and distance < 0.6:  # Umbral
                        best_distance = distance
                        best_match = self.known_names[i]
                
                recognized_names.append(best_match if best_match else "Desconocido")
                
            except Exception as e:
                print(f"Error en reconocimiento: {e}")
                recognized_names.append("Error")
        
        return faces, recognized_names
    
    def register_new_face(self, name, frame):
        """Registrar un nuevo rostro en la base de datos"""
        faces, gray = self.detector.detect_faces(frame)
        
        if len(faces) == 1:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            
            # Extraer embedding
            embedding = self.detector.extract_face_embedding(face_roi)
            
            # Serializar y guardar en base de datos
            embedding_blob = pickle.dumps(embedding)
            self.db.save_face(name, embedding_blob)
            
            # Recargar rostros conocidos
            self.load_known_faces()
            return True
        else:
            return False