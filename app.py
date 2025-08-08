# -*- coding: utf-8 -*-
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import tkinter as tk
from tkinter import messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
import os
import threading

FONT = ('Segoe UI', 10)

class SentioAIApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 MindCare Pro - Sistema de Apoyo Emocional")
        self.root.geometry("1400x850")

        self._configure_styles()

        model_path = os.path.join(os.path.dirname(__file__), 'modelo', 'modelo_emociones.h5')
        try:
            if os.path.exists(model_path):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                print("Modelo cargado exitosamente")
            else:
                raise FileNotFoundError(f"Archivo no encontrado en: {model_path}")
        except Exception as e:
            print(f"Error cargando modelo: {str(e)}")
            self.model = None
            if not messagebox.askyesno("Advertencia", "Modelo no encontrado. ¿Continuar en modo simulación?"):
                self.root.quit()
                return

        self._setup_ui()
        self.cap = None
        self.camera_thread = None
        self.running = False

    def _configure_styles(self):
        pass  # ttkbootstrap aplica estilos automáticamente

    def _setup_ui(self):
        main_frame = tb.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tb.Frame(main_frame, width=420, padding=10, bootstyle="light")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)

        right_frame = tb.Frame(main_frame, padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tb.Label(left_frame, text="💬 Chat de Apoyo Emocional", font=('Segoe UI', 13, 'bold')).pack(anchor=tk.W, pady=(0, 10))

        self.chat_history = tk.Text(left_frame, height=30, width=50, state=tk.DISABLED, wrap=tk.WORD, font=FONT,
                                    bg='white', fg='black', relief=tk.FLAT)
        self.chat_history.pack(fill=tk.BOTH, expand=True)

        input_frame = tb.Frame(left_frame)
        input_frame.pack(fill=tk.X, pady=5)

        self.user_input = tb.Entry(input_frame, font=FONT)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.user_input.bind("<Return>", lambda e: self.process_message())

        tb.Button(input_frame, text="Enviar", command=self.process_message, bootstyle="primary").pack(side=tk.LEFT)

        tb.Label(right_frame, text="📊 Análisis de Emociones en Tiempo Real", font=('Segoe UI', 13, 'bold')).pack(anchor=tk.W)

        self.fig, self.ax = plt.subplots(figsize=(6.5, 4.2), dpi=100)
        self.ax.set_ylim(0, 1)
        self.ax.grid(True, linestyle='--', alpha=0.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        controls_frame = tb.Frame(right_frame)
        controls_frame.pack(fill=tk.X, pady=10)
        tb.Button(controls_frame, text="🎥 Iniciar Cámara", command=self.start_camera, bootstyle="success").pack(side=tk.LEFT, padx=5)
        tb.Button(controls_frame, text="⛔ Detener Cámara", command=self.stop_camera, bootstyle="danger").pack(side=tk.LEFT, padx=5)

        self.camera_label = tb.Label(right_frame)
        self.camera_label.pack(pady=10)

        self.update_graph(['Feliz', 'Triste', 'Enojado', 'Neutral'], [0.25]*4)

    def process_message(self):
        message = self.user_input.get()
        if message.strip():
            self.display_message(f"Tú: {message}", "user")
            emotion, confidence = self.analyze_text(message)
            self.display_message(f"MindCare: Detecto {emotion} (Confianza: {confidence:.0%})", "bot")
            self.user_input.delete(0, tk.END)

    def analyze_text(self, text):
        emotions = ['Feliz', 'Triste', 'Enojado', 'Neutral']
        prob = np.random.dirichlet(np.ones(4), size=1)[0]
        self.update_graph(emotions, prob)
        return emotions[np.argmax(prob)], max(prob)

    def update_graph(self, emotions, values):
        self.ax.clear()
        colors = ['#27AE60', '#2980B9', '#E74C3C', '#95A5A6', '#8E44AD', '#F39C12', '#1ABC9C']
        bars = self.ax.bar(emotions, values, color=colors[:len(emotions)], width=0.5)
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.0%}', ha='center', va='bottom', fontsize=9)
        self.ax.set_title("Distribución de Emociones", pad=20)
        self.ax.set_ylim(0, 1.1)
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.canvas.draw()

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.running = True
                self.camera_thread = threading.Thread(target=self.show_camera_feed)
                self.camera_thread.daemon = True
                self.camera_thread.start()
            else:
                messagebox.showerror("Error", "No se pudo iniciar la cámara")
                self.cap.release()
                self.cap = None

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
            self.camera_label.config(image='')

    def show_camera_feed(self):
        logo_path = os.path.join(os.path.dirname(__file__), 'dcbadc76-f7ef-48a1-b831-8be4b7c498e1.png')
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is not None and logo.shape[2] == 4:
            alpha_logo = logo[:, :, 3] / 255.0
            logo = logo[:, :, :3]
        else:
            alpha_logo = None

        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray, (48, 48)) / 255.0
            roi = np.expand_dims(roi, axis=(0, -1))

            if self.model:
                try:
                    prediction = self.model.predict(roi, verbose=0)[0]
                    emotions = ['Enojado', 'Disgusto', 'Miedo', 'Feliz', 'Triste', 'Sorpresa', 'Neutral']
                    detected = emotions[np.argmax(prediction)]
                    emotion_probs = prediction
                    self.root.after(0, lambda: self.update_graph(emotions, emotion_probs))
                except Exception as e:
                    detected = "Error en modelo"
            else:
                detected = "Modo Simulación"

            cv2.putText(frame, f"{detected}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            if logo is not None:
                h, w, _ = logo.shape
                overlay = frame.copy()
                if alpha_logo is not None:
                    for c in range(3):
                        overlay[10:10+h, 10:10+w, c] = (alpha_logo * logo[:, :, c] +
                                                        (1 - alpha_logo) * overlay[10:10+h, 10:10+w, c])
                else:
                    overlay[10:10+h, 10:10+w] = logo
                frame = overlay

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.root.after(0, lambda img=img_tk: self.update_camera_image(img))

    def update_camera_image(self, img):
        self.camera_label.config(image=img)
        self.camera_label.image = img

    def display_message(self, message, sender):
        self.chat_history.config(state=tk.NORMAL)
        tag = "user" if sender == "user" else "bot"
        self.chat_history.insert(tk.END, message + "\n", tag)
        self.chat_history.tag_config("user", foreground="#3498DB", font=(FONT[0], 10, 'bold'))
        self.chat_history.tag_config("bot", foreground="#2C3E50", font=FONT)
        self.chat_history.see(tk.END)
        self.chat_history.config(state=tk.DISABLED)
        
if __name__ == "__main__":
    root = tb.Window(themename="morph")  # Puedes probar "flatly", "darkly", "minty", etc.
    app = SentioAIApp(root)
    root.mainloop()
