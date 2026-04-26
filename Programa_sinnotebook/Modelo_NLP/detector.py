#Librerias utilizadas

#pickle: Se utiliza para guardar y cargar objetos de Python, como modelos entrenados, de manera eficiente.
import pickle
#spacy: Es una biblioteca de procesamiento de lenguaje natural (NLP) que proporciona herramientas.
import spacy
#re: Se utiliza para trabajar con expresiones regulares, lo que es útil para limpiar y preprocesar texto.
import re
#scipy.sparse.hstack: Se utiliza para combinar matrices dispersas horizontalmente, lo que es común en la representación de texto con técnicas como TF-IDF.
from scipy.sparse import hstack
#numpy: Es una biblioteca fundamental para el cálculo numérico en Python, que se utiliza para manejar arrays y realizar operaciones matemáticas.
import numpy as np
#os: Se utiliza para interactuar con el sistema operativo, como manejar rutas de archivos y directorios.
import os

#Se carga el modelo de lenguaje de spaCy para español, que se utilizará para procesar y analizar el texto de los mensajes.
nlp = spacy.load("es_core_news_sm")
#Se define la ruta base del proyecto, que se utiliza para cargar los archivos de modelo y vectorizador guardados previamente.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#Se cargan el vectorizador y el modelo SVM previamente entrenados desde archivos pickle, lo que permite reutilizar estos objetos sin tener que volver a entrenarlos cada vez que se ejecute el código.
with open(os.path.join(BASE_DIR, "vectorizador.pkl"), "rb") as f:
    vectorizador = pickle.load(f)
with open(os.path.join(BASE_DIR, "modelo_svm.pkl"), "rb") as f:
    modelo_svm = pickle.load(f)


#Proceso de limpieza de texto
def limpiar_texto(texto):
    #Se realiza una limipieza del texto, como:
    #Convierte el texto a minúsculas.
    texto = str(texto).lower()
    #Elimina URLs y reemplaza por un token específico para mantener la información de que había una URL sin conservar la URL en sí.
    texto = re.sub(r'(https?://\S+|www\.\S+)', ' tokenurl ', texto)
    #Elimina direcciones de correo electrónico y reemplaza por un token específico para mantener la información de que había un correo sin conservar el correo en sí.
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)
    #Reemplaza números grandes, medianos y pequeños por tokens específicos para mantener la información de que había números sin conservar los números en sí.
    texto = re.sub(r"\b\d{5,}\b", " tokennumero_grande ", texto)
    texto = re.sub(r"\b\d{2,4}\b", " tokennumero_medio ", texto)
    texto = re.sub(r"\b\d\b", " tokennumero_pequeno ", texto)
    #texto = re.sub(r"[@#]\w+", " ", texto)
    #ACTUALIZADO: En lugar de eliminar completamente las palabras que comienzan con @ o #, solo eliminamos los símbolos pero mantenemos las palabras.
    #Solo borra los símbolos @ y #, pero deja las palabras intactas
    texto = re.sub(r"[@#]", " ", texto)
    #Elimina caracteres especiales, pero permite conservar los signos de exclamación, el símbolo de dólar y los espacios, ya que pueden ser relevantes para la detección de spam.
    texto = re.sub(r"[^a-záéíóúüñ0-9!$ ]", " ", texto)
    #Elimina espacios adicionales
    doc = nlp(texto)
    #Agregamos 'and not token.is_stop' para filtrar las palabras vacías
    tokens = []
    for token in doc:
        if not token.is_space:
            tokens.append(token.text)
    #Returnamos el texto limpio y tokenizado como una cadena de texto unida por espacios, ya que el vectorizador TF-IDF espera texto en formato de cadena.
    return " ".join(tokens)


#Función para extraer características manuales del mensaje original (antes de limpiar)
#Esto lo usamos para capturar información que podría perderse en la limpieza, como la presencia de URLs, números, signos de exclamación, etc.
def extraer_features(texto):
    #Convertimos el texto a string por si acaso viene en otro formato (aunque debería ser string, esto es una medida de seguridad).
    texto = str(texto)
    #Calculamos la longitud total del mensaje, el número de palabras, el número de dígitos, 
    # la cantidad de URLs (basado en el token que usamos para reemplazar las URLs), 
    # la cantidad de signos de exclamación y la cantidad de símbolos de dólar.
    longitud = len(texto)
    num_palabras = len(texto.split())
    num_numeros = len(re.findall(r'\d', texto))
    num_urls = texto.count("tokenurl")
    num_exclamaciones = texto.count("!")
    num_dolar = texto.count("$")
    #Retornamos estas caracteristicas
    return [longitud, num_palabras, num_numeros, num_urls, num_exclamaciones, num_dolar]

#Funcion principal para detectar si un mensaje es spam o ham
def detectar_spam(mensajes_dict_list):
    #Creamos una lista vacía para almacenar los mensajes que el modelo clasifique como spam.
    mensajes_spam = []

    #Ciclo que recorre cada mensaje en la lista de mensajes.
    for msg in mensajes_dict_list:
        #Obtenemos el texto del mensaje utilizando la clave "mensaje". Si la clave no existe, se asigna una cadena vacía para evitar errores.
        texto = msg.get("mensaje", "")
        #Limpiamos el texto del mensaje utilizando la función limpiar_texto, lo que prepara el texto para ser procesado por el modelo.
        texto_limpio = limpiar_texto(texto)
        #Transformamos el texto limpio en una representación numérica utilizando el vectorizador TF-IDF previamente cargado. 
        # Esto convierte el texto en una matriz dispersa de características que el modelo SVM puede procesar.
        tfidf = vectorizador.transform([texto_limpio])
        #Extraemos las características manuales del mensaje original (antes de limpiar) utilizando la función extraer_features, lo que nos da un array con las características adicionales que hemos definido.
        features_extra = np.array([extraer_features(texto)])
        #Combinamos la representación TF-IDF con las características manuales utilizando hstack para crear un vector final que contenga toda la información relevante para la clasificación.
        vector_final = hstack([tfidf, features_extra])
        #El modelo SVM hace una predicción utilizando el vector final, lo que devuelve una etiqueta de clase ('spam' o 'ham') para el mensaje.
        prediccion = modelo_svm.predict(vector_final)[0]

        #Si es spam agregamos el mensaje original (sin limpiar) a la lista de mensajes_spam, lo que nos permite conservar el mensaje tal como fue recibido para su posterior análisis o uso.
        if prediccion == 'spam':
            mensajes_spam.append(msg)

    #Finalmente, la función retorna la lista de mensajes que han sido clasificados como spam.
    return mensajes_spam