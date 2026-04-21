import pickle
import spacy
import re
import os
#python -m spacy download es_core_news_sm
nlp = spacy.load("es_core_news_sm")
stopwords = nlp.Defaults.stop_words

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "vectorizador.pkl"), "rb") as f:
    vectorizador = pickle.load(f)

# Volvemos a cargar el modelo SVM
with open(os.path.join(BASE_DIR, "modelo_svm.pkl"), "rb") as f:
    modelo_svm = pickle.load(f)

def limpiar_texto(texto):
    texto = str(texto).lower()
    #texto = re.sub(r"http\S+|www\.\S+", " tokenurl ", texto)
    # En detector.py reemplaza la línea del http por esta:
    texto = re.sub(r'(?:https?://\S+|(?:www\.)?bit\.ly/\S+|\b[\w.-]+\.(?:com|net|mx)\b(?:/\S*)?)', ' tokenurl ', texto)
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)
    texto = re.sub(r"\d+", " tokennumero ", texto)
    texto = re.sub(r"[@#]\w+", " ", texto)
    texto = re.sub(r"[^\w\sáéíóúüñ]", " ", texto, flags=re.IGNORECASE)
    
    doc = nlp(texto)
    lemas = []
    for token in doc:
        if not token.is_space and token.lemma_ not in stopwords:
            lemas.append(token.lemma_)
                
    return " ".join(lemas)

def detectar_spam(mensajes_dict_list):
    mensajes_spam = []

    for msg in mensajes_dict_list:
        texto_limpio = limpiar_texto(msg['mensaje'])
        vector = vectorizador.transform([texto_limpio])
        # Usamos el modelo SVM
        prediccion = modelo_svm.predict(vector)[0]

        if prediccion == 'spam':
            mensajes_spam.append(msg)

    return mensajes_spam