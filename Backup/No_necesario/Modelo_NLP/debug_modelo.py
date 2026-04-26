import spacy
import re
import pickle


def limpiar_y_lematizar(texto):
    texto = str(texto).lower()
    
    #Ya no eliminamos los números, sino que los reemplazamos por un token específico para que el modelo aprenda a reconocer su presencia sin memorizar cada número específico.
    texto = re.sub(r"http\S+|www\.\S+", " tokenurl ", texto)
    #texto = re.sub(r"bit.ly\S+|/\S+", " tokenurl_bitly ", texto) 
    #texto = re.sub(r'(?:https?://\S+|(?:www.)?bit.ly/\S+|\b[\w.-]+.(?:com|net)\b(?:/\S*)?)', ' tokenurl ', texto)
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


# 1. CARGAMOS TUS HERRAMIENTAS YA ENTRENADAS
nlp = spacy.load("es_core_news_sm")
stopwords = nlp.Defaults.stop_words

# (AQUÍ DEBES PEGAR TU FUNCIÓN limpiar_y_lematizar EXACTAMENTE COMO LA TIENES)

with open("Modelo_NLP/vectorizador.pkl", "rb") as f:
    vectorizador = pickle.load(f)

with open("Modelo_NLP/modelo_svm.pkl", "rb") as f:
    modelo_svm = pickle.load(f)

# 2. EL MENSAJE REBELDE
mensaje_rebelde = "Tu paquete de MercadoLibre no pudo ser entregado. Actualiza aqui: http://correos-falsos.com"

# 3. RADIOGRAFÍA
print("\n--- RADIOGRAFÍA DEL MENSAJE ---")
mensaje_limpio = limpiar_y_lematizar(mensaje_rebelde)
print(f"1. Después de limpiar y lematizar:\n'{mensaje_limpio}'\n")

# Transformamos a números
vector = vectorizador.transform([mensaje_limpio])

# Invertimos la transformación para ver qué palabras sobrevivieron al min_df=5
palabras_reconocidas = vectorizador.inverse_transform(vector)[0]
print(f"2. Palabras que el modelo REALMENTE reconoce de este mensaje (su diccionario):")
print(palabras_reconocidas)
print("\nSi esta lista está vacía o solo tiene palabras inofensivas, ¡por eso falla!")

# Vemos las probabilidades de la predicción
probabilidades = modelo_svm.predict_proba(vector)[0]
clases = modelo_svm.classes_ # Generalmente ['ham', 'spam']

print("\n3. Veredicto del SVM:")
for i, clase in enumerate(clases):
    print(f"Probabilidad de {clase}: {probabilidades[i] * 100:.2f}%")