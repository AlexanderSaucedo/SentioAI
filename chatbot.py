import pickle
import numpy as np
from tensorflow.python.keras.models  import load_model

class Chatbot:
    def __init__(self):
        # Cargar modelo NLP (ejemplo con TextBlob)
        from textblob import TextBlob
        self.nlp_model = TextBlob
        
        # Opcional: Cargar modelo CNN para imágenes
        self.cnn_model = load_model('C:\Users\PC\Desktop\SentioAI\modeloss\modelo_emociones_final.h5')
    
    def predecir_emocion(self, texto):
        # Ejemplo simple con TextBlob (mejorar con tu modelo NLP)
        analysis = self.nlp_model(texto).translate(to='en')
        polarity = analysis.sentiment.polarity
        
        if polarity > 0.3:
            return "alegría", 0.8
        elif polarity < -0.3:
            return "tristeza", 0.7
        else:
            return "neutral", 0.6