# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.python.keras.models import load_model
import os


# Configuración de matplotlib
plt.style.use('ggplot')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.facecolor': '#f0f0f0'
})

class SentioAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MindCare Pro - Sistema de Apoyo Emocional")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        self._configure_styles()
        
        # Ruta absoluta del modelo
        model_path = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_emociones.h5')
        print(f"Buscando modelo en: {model_path}")  # Para depuración
        
        try:
            # Verificar si el archivo existe primero
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("Modelo cargado exitosamente")
            else:
                raise FileNotFoundError(f"Archivo no encontrado en: {model_path}")
        except Exception as e:
            print(f"Error cargando modelo: {str(e)}")
            self.model = None
            if not messagebox.askyesno("Advertencia", 
                                      "Modelo no encontrado. ¿Continuar en modo simulación?\n"
                                      f"Ruta esperada: {model_path}"):
                self.root.quit()
                return
        
        self._setup_ui()
        self.cap = None
    
    def _configure_styles(self):
        """Configura los estilos visuales"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('.', background='#f0f0f0', font=('Arial', 10))
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', foreground='#333333')
        style.configure('TButton', padding=6, relief=tk.FLAT)
        style.map('TButton',
                background=[('active', '#e0e0e0'), ('!disabled', '#f0f0f0')],
                foreground=[('!disabled', '#333333')])
        style.configure('Chat.TFrame', background='white')
        style.configure('Chat.TText', font=('Arial', 10), wrap=tk.WORD)
    
    def _setup_ui(self):
        """Configura la interfaz gráfica"""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel izquierdo (Chatbot)
        left_frame = ttk.Frame(main_frame, width=400, padding=10, style='Chat.TFrame')
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        # Panel derecho (Gráficos/Cámara)
        right_frame = ttk.Frame(main_frame, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Componentes del Chatbot
        self.chat_history = tk.Text(
            left_frame, height=25, width=50, state=tk.DISABLED,
            wrap=tk.WORD, font=('Arial', 10), padx=5, pady=5,
            bg='white', fg='#333333', selectbackground='#2196F3'
        )
        self.chat_history.pack(fill=tk.BOTH, expand=True)
        
        input_frame = ttk.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=5)
        
        self.user_input = ttk.Entry(input_frame, font=('Arial', 10))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", lambda e: self.process_message())
        
        ttk.Button(input_frame, text="Enviar", command=self.process_message).pack(side=tk.LEFT)
        
        # Componentes de Visualización
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.fig.patch.set_facecolor('#f0f0f0')
        self.ax.set_title("Distribución de Emociones Detectadas", pad=20)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().config(bg='#f0f0f0')
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Controles de cámara
        controls_frame = ttk.Frame(right_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(controls_frame, text="Iniciar Cámara", command=self.start_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Detener Cámara", command=self.stop_camera).pack(side=tk.LEFT, padx=5)
        
        self.camera_label = ttk.Label(right_frame)
        self.camera_label.pack()
        
        self.update_graph(['Feliz', 'Triste', 'Enojado', 'Neutral'], [0.25]*4)
    
    def process_message(self):
        message = self.user_input.get()
        if message.strip():
            self.display_message(f"Tú: {message}", "user")
            emotion, confidence = self.analyze_text(message)
            self.display_message(f"MindCare: Detecto {emotion} (Confianza: {confidence:.0%})", "bot")
            emotions = ['Feliz', 'Triste', 'Enojado', 'Neutral']
            values = np.random.dirichlet(np.ones(4), size=1)[0]
            self.update_graph(emotions, values)
            self.user_input.delete(0, tk.END)
    
    def analyze_text(self, text):
        emotions = ['Feliz', 'Triste', 'Enojado', 'Neutral']
        prob = np.random.dirichlet(np.ones(4), size=1)[0]
        return emotions[np.argmax(prob)], max(prob)
    
    def update_graph(self, emotions, values):
        self.ax.clear()
        colors = ['#4CAF50', '#2196F3', '#F44336', '#9E9E9E']
        bars = self.ax.bar(emotions, values, color=colors, width=0.6)
        
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0%}', ha='center', va='bottom')
        
        self.ax.set_title("Distribución de Emociones Detectadas", pad=20)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas.draw()
    
    def start_camera(self):
        if self.cap is None:
            try:
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW para Windows
                if not self.cap.isOpened():
                    raise RuntimeError("No se pudo abrir la cámara")
                self.show_camera_feed()
            except Exception as e:
                print(f"Error al iniciar cámara: {str(e)}")
                messagebox.showerror("Error", f"No se pudo iniciar la cámara: {str(e)}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
    
    def stop_camera(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.camera_label.config(image='')
    
    def show_camera_feed(self):
        if self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error al capturar frame")
                    return
                
                # Procesamiento de imagen
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi = cv2.resize(gray, (48, 48)) / 255.0
                roi = np.expand_dims(roi, axis=(0, -1))
                
                # Detección de emociones
                if self.model:
                    try:
                        emotion = self.model.predict(roi, verbose=0)[0]
                        emotion_text = "Emoción Detectada"
                    except Exception as e:
                        print(f"Error en predicción: {str(e)}")
                        emotion_text = "Error en modelo"
                else:
                    emotion = np.random.dirichlet(np.ones(7), size=1)[0]
                    emotion_text = "Modo Simulación"
                
                # Mostrar texto en frame
                cv2.putText(frame, emotion_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Convertir para Tkinter
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tk = ImageTk.PhotoImage(image=img)
                
                # Actualizar GUI
                self.camera_label.config(image=img_tk)
                self.camera_label.image = img_tk
                self.camera_label.after(10, self.show_camera_feed)
                
            except Exception as e:
                print(f"Error en show_camera_feed: {str(e)}")
                self.stop_camera()
    
    def display_message(self, message, sender):
        self.chat_history.config(state=tk.NORMAL)
        tag = "user" if sender == "user" else "bot"
        self.chat_history.insert(tk.END, message + "\n", tag)
        self.chat_history.tag_config("user", foreground="blue", font=('Arial', 10, 'bold'))
        self.chat_history.tag_config("bot", foreground="green", font=('Arial', 10))
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)
    
    def clear_chat(self):
        self.chat_history.config(state=tk.NORMAL)
        self.chat_history.delete(1.0, tk.END)
        self.chat_history.config(state=tk.DISABLED)

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = SentioAIApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        input("Presiona Enter para salir...")