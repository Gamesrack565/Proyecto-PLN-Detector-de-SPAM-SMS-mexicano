import pandas as pd
import spacy
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("1. Cargando el modelo de lenguaje de Spacy (español)...")
nlp = spacy.load("es_core_news_sm")
stopwords = nlp.Defaults.stop_words

def limpiar_y_lematizar(texto):
    texto = str(texto).lower()
    
    # Conservamos pistas clave como Tokens
    texto = re.sub(r"http\S+|www\.\S+", " tokenurl ", texto) 
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
ruta_csv = "datos/spam_espanol.csv" # Ajusta tu ruta
df = pd.read_csv(ruta_csv)
df = df.dropna(subset=['mensaje', 'etiqueta'])

print(f"3. Limpiando y lematizando {len(df)} mensajes (Paciencia)...")
df['mensaje_limpio'] = df['mensaje'].apply(limpiar_y_lematizar)

X = df['mensaje_limpio']
y = df['etiqueta']

X_entrenar, X_examen, y_entrenar, y_examen = train_test_split(X, y, test_size=0.2, random_state=42)

print("4. Vectorizando con TF-IDF...")
# CAMBIO CRÍTICO: min_df=1 permite que palabras raras como "bbva" o "cajero" NO se borren.
vectorizador = TfidfVectorizer(max_features=2500, min_df=1, ngram_range=(1, 2))
X_entrenar_tfidf = vectorizador.fit_transform(X_entrenar)
X_examen_tfidf = vectorizador.transform(X_examen)

print("5. Entrenando el Bosque Aleatorio (Anti-Sobreajuste)...")
# CAMBIO CRÍTICO: max_depth y min_samples_leaf limitan la memoria del modelo.
modelo_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=30,           # Le prohíbe memorizar (crear ramas muy profundas)
    min_samples_leaf=2,     # Exige patrones generales
    class_weight='balanced',# Le da más importancia al Spam
    random_state=42
)
modelo_rf.fit(X_entrenar_tfidf, y_entrenar)

print("6. Evaluando el modelo...")
y_pred = modelo_rf.predict(X_examen_tfidf)
precision = accuracy_score(y_examen, y_pred)
print(f"\n¡Precisión general (Realista): {precision * 100:.2f}%\n")
print("Reporte detallado de clasificación:")
print(classification_report(y_examen, y_pred))

print("7. Guardando el 'Cerebro'...")
with open("vectorizador.pkl", "wb") as f:
    pickle.dump(vectorizador, f)

# Lo guardamos con un nombre nuevo
with open("modelo_bosque.pkl", "wb") as f:
    pickle.dump(modelo_rf, f)

print("Archivos actualizados correctamente.")