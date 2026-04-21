import pickle
import spacy
import re
import os

# 1. Cargar Spacy y Stopwords
nlp = spacy.load("es_core_news_sm")
stopwords = nlp.Defaults.stop_words

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizador.pkl"), "rb") as f:
    vectorizador = pickle.load(f)

with open(os.path.join(BASE_DIR, "modelo_svm.pkl"), "rb") as f:
    modelo_svm = pickle.load(f)

def limpiar_texto(texto):
    """Misma limpieza Anti-Sobreajuste que en el entrenamiento"""
    texto = str(texto).lower()
    
    # 1. Conservar pistas clave como Tokens
    texto = re.sub(r"http\S+|www\.\S+", " tokenurl ", texto)
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)
    texto = re.sub(r"\d+", " tokennumero ", texto)
    
    # 2. Eliminar menciones, puntuación y caracteres raros
    texto = re.sub(r"[@#]\w+", " ", texto)
    texto = re.sub(r"[^\w\sáéíóúüñ]", " ", texto, flags=re.IGNORECASE)
    
    doc = nlp(texto)
    lemas = []
    for token in doc:
        if not token.is_space and token.lemma_ not in stopwords:
            lemas.append(token.lemma_)
                
    return " ".join(lemas)

def detectar_spam(mensajes_dict_list):
    """
    Recibe los mensajes de Android, los limpia y los pasa por el SVM.
    Devuelve ÚNICAMENTE los mensajes que el SVM haya clasificado como 'spam'.
    """
    mensajes_spam = []

    for msg in mensajes_dict_list:
        texto_limpio = limpiar_texto(msg['mensaje'])
        vector = vectorizador.transform([texto_limpio])
        prediccion = modelo_svm.predict(vector)[0]

        if prediccion == 'spam':
            mensajes_spam.append(msg)

    return mensajes_spam