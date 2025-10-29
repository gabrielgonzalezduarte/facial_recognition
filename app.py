import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import cv2
from PIL import Image, ImageTk
import threading
import time
from face_recognition import FaceRecognizer

class FacialRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconocimiento Facial")
        self.root.geometry("800x600")
        
        # Inicializar reconocedor facial
        self.recognizer = FaceRecognizer()
        self.is_camera_active = False
        self.cap = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Frame de controles
        control_frame = ttk.LabelFrame(main_frame, text="Controles", padding="5")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Botones de control
        ttk.Button(control_frame, text="Iniciar Cámara", 
                  command=self.start_camera).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Detener Cámara", 
                  command=self.stop_camera).grid(row=0, column=1, padx=5)
        ttk.Button(control_frame, text="Registrar Rostro", 
                  command=self.register_face).grid(row=0, column=2, padx=5)
        ttk.Button(control_frame, text="Capturar Foto", 
                  command=self.capture_photo).grid(row=0, column=3, padx=5)
        
        # Frame para registro de rostros
        register_frame = ttk.LabelFrame(main_frame, text="Registrar Nuevo Rostro", padding="5")
        register_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        ttk.Label(register_frame, text="Nombre:").grid(row=0, column=0, sticky=tk.W)
        self.name_entry = ttk.Entry(register_frame, width=20)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(register_frame, text="Registrar", 
                  command=self.register_from_entry).grid(row=1, column=0, columnspan=2, pady=5)
        
        # Frame para lista de rostros registrados
        faces_frame = ttk.LabelFrame(main_frame, text="Rostros Registrados", padding="5")
        faces_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Lista de rostros
        self.faces_listbox = tk.Listbox(faces_frame, height=8)
        self.faces_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(faces_frame, orient=tk.VERTICAL, command=self.faces_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.faces_listbox.configure(yscrollcommand=scrollbar.set)
        
        ttk.Button(faces_frame, text="Eliminar Seleccionado", 
                  command=self.delete_face).grid(row=1, column=0, pady=5)
        
        ttk.Button(faces_frame, text="Actualizar Lista", 
                  command=self.update_faces_list).grid(row=2, column=0, pady=5)
        
        # Frame para video
        video_frame = ttk.LabelFrame(main_frame, text="Vista de Cámara", padding="5")
        video_frame.grid(row=0, column=1, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        
        # Label para mostrar video
        self.video_label = ttk.Label(video_frame, text="Cámara no iniciada")
        self.video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar weights para expansión
        main_frame.rowconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.columnconfigure(1, weight=1)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)
        register_frame.columnconfigure(1, weight=1)
        faces_frame.columnconfigure(0, weight=1)
        faces_frame.rowconfigure(0, weight=1)
        
        # Actualizar lista inicial
        self.update_faces_list()
    
    def start_camera(self):
        if not self.is_camera_active:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "No se pudo acceder a la cámara")
                return
            
            self.is_camera_active = True
            self.video_thread = threading.Thread(target=self.update_video, daemon=True)
            self.video_thread.start()
    
    def stop_camera(self):
        self.is_camera_active = False
        if self.cap:
            self.cap.release()
            self.video_label.configure(text="Cámara detenida")
    
    def update_video(self):
        while self.is_camera_active and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Realizar reconocimiento facial
                faces, names = self.recognizer.recognize_face(frame)
                
                # Dibujar rostros detectados
                for i, (face, name) in enumerate(zip(faces, names)):
                    if name:
                        self.recognizer.detector.draw_faces(frame, [face], name)
                    else:
                        self.recognizer.detector.draw_faces(frame, [face])
                
                # Convertir a formato Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Actualizar interfaz
                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk
            
            time.sleep(0.03)  # ~30 FPS
    
    def register_face(self):
        if not self.is_camera_active:
            messagebox.showwarning("Advertencia", "Inicie la cámara primero")
            return
        
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Advertencia", "Ingrese un nombre")
            return
        
        ret, frame = self.cap.read()
        if ret:
            success = self.recognizer.register_new_face(name, frame)
            if success:
                messagebox.showinfo("Éxito", f"Rostro de {name} registrado correctamente")
                self.update_faces_list()
                self.name_entry.delete(0, tk.END)
            else:
                messagebox.showerror("Error", "No se pudo detectar un rostro único")
    
    def register_from_entry(self):
        self.register_face()
    
    def capture_photo(self):
        if not self.is_camera_active:
            messagebox.showwarning("Advertencia", "Inicie la cámara primero")
            return
        
        ret, frame = self.cap.read()
        if ret:
            filename = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
            )
            if filename:
                cv2.imwrite(filename, frame)
                messagebox.showinfo("Éxito", f"Foto guardada como {filename}")
    
    def update_faces_list(self):
        self.faces_listbox.delete(0, tk.END)
        faces_data = self.recognizer.db.get_all_faces()
        for face_id, name, _ in faces_data:
            self.faces_listbox.insert(tk.END, f"{name} (ID: {face_id})")
    
    def delete_face(self):
        selection = self.faces_listbox.curselection()
        if selection:
            item_text = self.faces_listbox.get(selection[0])
            # Extraer ID del texto
            try:
                face_id = int(item_text.split("ID: ")[1].strip(")"))
                self.recognizer.db.delete_face(face_id)
                self.recognizer.load_known_faces()
                self.update_faces_list()
                messagebox.showinfo("Éxito", "Rostro eliminado")
            except:
                messagebox.showerror("Error", "No se pudo eliminar el rostro")
        else:
            messagebox.showwarning("Advertencia", "Seleccione un rostro para eliminar")
    
    def __del__(self):
        self.stop_camera()
        if hasattr(self, 'recognizer'):
            self.recognizer.db.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = FacialRecognitionApp(root)
    root.mainloop()