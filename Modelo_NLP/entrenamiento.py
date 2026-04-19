import pandas as pd
import spacy
import re
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("1. Cargando el modelo de lenguaje de Spacy (español)...")
nlp = spacy.load("es_core_news_sm")
stopwords = nlp.Defaults.stop_words

def limpiar_y_lematizar(texto):
    texto = str(texto).lower()
    
    #Ya no eliminamos los números, sino que los reemplazamos por un token específico para que el modelo aprenda a reconocer su presencia sin memorizar cada número específico.
    #texto = re.sub(r"http\S+|www\.\S+", " tokenurl ", texto)
    #texto = re.sub(r"bit.ly\S+|/\S+", " tokenurl_bitly ", texto) 
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

print("2. Leyendo el dataset...")
#Dataset actualizado con mensajes traducidos a español y mensajes enfocados en mexico
ruta_csv = "datos/spam_espanol.csv"
df = pd.read_csv(ruta_csv)
#Eliminamos filas con mensajes o etiquetas vacías para evitar problemas en el entrenamiento
df = df.dropna(subset=['mensaje', 'etiqueta'])


print(f"3. Limpiando y lematizando {len(df)} mensajes...")
#Se realiza el proceso de limpieza y lematización en la columna 'mensaje' y se guarda el resultado en una nueva columna 'mensaje_limpio' para mantener el original intacto.
df['mensaje_limpio'] = df['mensaje'].apply(limpiar_y_lematizar)
X = df['mensaje_limpio']
y = df['etiqueta']

#Dividimos el dataset en entrenamiento y examen (80% entrenamiento, 20% examen)
#AUN NO SE HACE EL APRENDIZAJE SOLO LO DIVIDIO
X_entrenar, X_examen, y_entrenar, y_examen = train_test_split(X, y, test_size=0.2)

print("4. Vectorizando con TF-IDF...")
#Bajamos a 800 palabras máximo para que no memorice todo el diccionario
#Se reduce el rango de n-gramas a (1,1) para enfocarnos solo en palabras individuales y evitar combinaciones que podrían llevar a sobreajuste.
#El parámetro min_df=1 asegura que se incluyan todas las palabras que aparecen al menos una vez, lo que es importante para un dataset pequeño y evitar eliminar palabras potencialmente útiles.
vectorizador = TfidfVectorizer(max_features=2500, min_df=2, ngram_range=(1, 2))

#VECTORIZA LOS MENSAJES QUE FUERON ELEGIDOS PARA ENTRENAR Y PARA EXAMEN 
X_entrenar_tfidf = vectorizador.fit_transform(X_entrenar)
X_examen_tfidf = vectorizador.transform(X_examen)

print("5. Entrenando el SVM...")
#Preparacion de entrenamiento del modelo SVM con kernel lineal, regularización C=0.1, probabilidad habilitada para obtener scores de confianza, y class_weight='balanced' para manejar cualquier desequilibrio en las clases.
modelo_svm = SVC(kernel='linear', C=0.1, probability=True, class_weight='balanced')
#Ingreso de variables para el modelo
modelo_svm.fit(X_entrenar_tfidf, y_entrenar)


# Evaluando entrenamiento vs examen
y_pred_entrenar = modelo_svm.predict(X_entrenar_tfidf)
precision_entrenar = accuracy_score(y_entrenar, y_pred_entrenar)
print(f"Precisión en Entrenamiento: {precision_entrenar * 100:.2f}%")


print("6. Evaluando el modelo...")
#Proceso de evaluacion del modelo utilizando el conjunto de examen. Se predicen las etiquetas para los mensajes de examen y se calcula la precisión general, así como un reporte detallado de clasificación que incluye métricas como precisión, recall y F1-score para cada clase.
y_pred = modelo_svm.predict(X_examen_tfidf)
#Se obtiene la precision general del modelo 
precision = accuracy_score(y_examen, y_pred)
print(f"\n¡Precisión general: {precision * 100:.2f}%\n")
print("Reporte detallado de clasificación:")
print(classification_report(y_examen, y_pred))

print(f"\n BRECHA DE PRECISIÓN ENTRE ENTRENAMIENTO Y EXAMEN: {(precision_entrenar - precision) * 100:.2f}%")

print("7. Guardando...")
carpeta_modelo = "Modelo_NLP"
os.makedirs(carpeta_modelo, exist_ok=True)
with open(f"{carpeta_modelo}/vectorizador.pkl", "wb") as f:
    pickle.dump(vectorizador, f)

with open(f"{carpeta_modelo}/modelo_svm.pkl", "wb") as f:
    pickle.dump(modelo_svm, f)

print("Archivos actualizados correctamente.")