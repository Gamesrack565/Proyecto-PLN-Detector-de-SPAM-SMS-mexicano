import pandas as pd
import spacy
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

print("1. Cargando el modelo de lenguaje de Spacy (español)...")
nlp = spacy.load("es_core_news_sm")
stopwords = nlp.Defaults.stop_words

def limpiar_y_lematizar(texto):
    """
    Limpieza de texto Anti-Sobreajuste:
    En lugar de borrar números y enlaces, los convertimos en 'tokens' clave.
    """
    texto = str(texto).lower()
    
    # 1. Conservar pistas clave como Tokens
    texto = re.sub(r"http\S+|www\.\S+", " tokenurl ", texto) # Los links ahora son 'tokenurl'
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)  # Los correos son 'tokencorreo'
    texto = re.sub(r"\d+", " tokennumero ", texto)           # Los números son 'tokennumero'
    
    # 2. Eliminar menciones, puntuación y caracteres raros
    texto = re.sub(r"[@#]\w+", " ", texto)
    texto = re.sub(r"[^\w\sáéíóúüñ]", " ", texto, flags=re.IGNORECASE)
    
    # 3. Tokenizar, lematizar y quitar stop-words
    doc = nlp(texto)
    lemas = []
    for token in doc:
        if not token.is_space and token.lemma_ not in stopwords:
            # Quitamos la restricción estricta de .isalpha() para que pasen nuestros 'tokens'
            lemas.append(token.lemma_)
                
    return " ".join(lemas)

print("2. Leyendo el dataset traducido...")
ruta_csv = "datos/spam_espanol.csv"
df = pd.read_csv(ruta_csv)
df = df.dropna(subset=['mensaje', 'etiqueta'])

print(f"3. Limpiando y lematizando {len(df)} mensajes (Paciencia)...")
df['mensaje_limpio'] = df['mensaje'].apply(limpiar_y_lematizar)

print("4. Dividiendo los datos (80% entrenamiento, 20% prueba)...")
X = df['mensaje_limpio']
y = df['etiqueta']

X_entrenar, X_examen, y_entrenar, y_examen = train_test_split(X, y, test_size=0.2, random_state=42)

print("5. Vectorizando con TF-IDF (Anti-Sobreajuste)...")
# max_features=3000: Evita memorizar palabras raras (solo usa las 3000 más importantes).
# min_df=3: Ignora palabras que aparecen en menos de 3 mensajes.
# ngram_range=(1,2): Aprende combinaciones de 2 palabras (ej. "ganaste tokennumero").
vectorizador = TfidfVectorizer(max_features=3000, min_df=3, ngram_range=(1, 2))
X_entrenar_tfidf = vectorizador.fit_transform(X_entrenar)
X_examen_tfidf = vectorizador.transform(X_examen)

print("6. Entrenando el modelo SVM (Balanceado)...")
# class_weight='balanced': Le da más importancia matemática a los pocos SPAM que tenemos.
modelo_svm = SVC(kernel='linear', probability=True, class_weight='balanced')
modelo_svm.fit(X_entrenar_tfidf, y_entrenar)

print("7. Evaluando el modelo...")
y_pred = modelo_svm.predict(X_examen_tfidf)
precision = accuracy_score(y_examen, y_pred)
print(f"\n¡Entrenamiento completado! Precisión general: {precision * 100:.2f}%\n")
print("Reporte detallado de clasificación:")
print(classification_report(y_examen, y_pred))

print("8. Guardando el 'Cerebro'...")
with open("vectorizador.pkl", "wb") as f:
    pickle.dump(vectorizador, f)

with open("modelo_svm.pkl", "wb") as f:
    pickle.dump(modelo_svm, f)

print("Archivos actualizados correctamente.")