import pickle
import spacy
import re
from scipy.sparse import hstack
import numpy as np
import os

nlp = spacy.load("es_core_news_sm")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizador.pkl"), "rb") as f:
    vectorizador = pickle.load(f)

with open(os.path.join(BASE_DIR, "modelo_svm.pkl"), "rb") as f:
    modelo_svm = pickle.load(f)


# =========================
# MISMA LIMPIEZA QUE ENTRENAMIENTO
# =========================
def limpiar_texto(texto):
    texto = str(texto).lower()

    texto = re.sub(r'(https?://\S+|www\.\S+)', ' tokenurl ', texto)
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)

    texto = re.sub(r"\b\d{5,}\b", " tokennumero_grande ", texto)
    texto = re.sub(r"\b\d{2,4}\b", " tokennumero_medio ", texto)
    texto = re.sub(r"\b\d\b", " tokennumero_pequeno ", texto)

    texto = re.sub(r"[@#]\w+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ0-9!$ ]", " ", texto)

    doc = nlp(texto)

    tokens = []
    for token in doc:
        if not token.is_space:
            tokens.append(token.text)

    return " ".join(tokens)


# =========================
# MISMAS FEATURES
# =========================
def extraer_features(texto):
    texto = str(texto)

    longitud = len(texto)
    num_palabras = len(texto.split())
    num_numeros = len(re.findall(r'\d', texto))
    num_urls = texto.count("tokenurl")
    num_exclamaciones = texto.count("!")
    num_dolar = texto.count("$")

    return [longitud, num_palabras, num_numeros, num_urls, num_exclamaciones, num_dolar]


# =========================
# FUNCIÓN PRINCIPAL
# =========================
def detectar_spam(mensajes_dict_list):
    mensajes_spam = []

    for msg in mensajes_dict_list:
        texto = msg.get("mensaje", "")

        texto_limpio = limpiar_texto(texto)

        tfidf = vectorizador.transform([texto_limpio])
        features_extra = np.array([extraer_features(texto)])

        vector_final = hstack([tfidf, features_extra])

        prediccion = modelo_svm.predict(vector_final)[0]

        if prediccion == 'spam':
            mensajes_spam.append(msg)

    return mensajes_spam