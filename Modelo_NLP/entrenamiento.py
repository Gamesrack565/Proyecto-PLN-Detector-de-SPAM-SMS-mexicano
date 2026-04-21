import pandas as pd
import spacy
import re
import pickle
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

print("1. Cargando el modelo de lenguaje de Spacy (español)...")
nlp = spacy.load("es_core_news_sm")

# =========================
# LIMPIEZA MEJORADA
# =========================
def limpiar_texto(texto):
    texto = str(texto).lower()

    # URLs
    texto = re.sub(r'(https?://\S+|www\.\S+)', ' tokenurl ', texto)

    # Correos
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)

    # Números por rangos (mejor que uno solo)
    texto = re.sub(r"\b\d{5,}\b", " tokennumero_grande ", texto)
    texto = re.sub(r"\b\d{2,4}\b", " tokennumero_medio ", texto)
    texto = re.sub(r"\b\d\b", " tokennumero_pequeno ", texto)

    # Menciones
    texto = re.sub(r"[@#]\w+", " ", texto)

    # Mantener solo letras, números y signos útiles
    texto = re.sub(r"[^a-záéíóúüñ0-9!$ ]", " ", texto)

    doc = nlp(texto)

    tokens = []
    for token in doc:
        if not token.is_space:
            tokens.append(token.text)

    return " ".join(tokens)


# =========================
# FEATURES ADICIONALES
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


print("2. Leyendo el dataset...")
ruta_csv = "datos/spam_espanol.csv"
df = pd.read_csv(ruta_csv)

df = df.dropna(subset=['mensaje', 'etiqueta'])

print(f"3. Limpiando {len(df)} mensajes...")
df['mensaje_limpio'] = df['mensaje'].apply(limpiar_texto)

# Eliminar vacíos
df = df[df['mensaje_limpio'].str.strip() != ""]

print("=== ANÁLISIS DE DUPLICADOS ===")
print(f"Mensajes totales: {len(df)}")

duplicados = df.duplicated(subset=['mensaje_limpio']).sum()
print(f"Duplicados exactos: {duplicados}")

# Eliminar duplicados exactos
df = df.drop_duplicates(subset=['mensaje_limpio'])

# =========================
# ELIMINAR DUPLICADOS SEMÁNTICOS
# =========================
print("Eliminando duplicados semánticos (esto puede tardar)...")

vector_temp = TfidfVectorizer(max_features=3000)
X_temp = vector_temp.fit_transform(df['mensaje_limpio'])

sim_matrix = cosine_similarity(X_temp)

to_remove = set()
threshold = 0.90

for i in range(sim_matrix.shape[0]):
    for j in range(i + 1, sim_matrix.shape[1]):
        if sim_matrix[i, j] > threshold:
            to_remove.add(j)

df = df.drop(df.index[list(to_remove)])

print(f"Mensajes después de eliminar duplicados semánticos: {len(df)}")

# =========================
# PREPARACIÓN
# =========================
X = df['mensaje_limpio']
y = df['etiqueta']

# Features adicionales
features_extra = np.array([extraer_features(t) for t in df['mensaje']])

# Split
X_entrenar, X_examen, y_entrenar, y_examen, feat_train, feat_test = train_test_split(
    X, y, features_extra, test_size=0.2, stratify=y
)

print("\n--- Muestras de ENTRENAMIENTO ---")
print(X_entrenar.head(3))

print("\n--- Muestras de EXAMEN ---")
print(X_examen.head(3))

# =========================
# TF-IDF MEJORADO
# =========================
print("4. Vectorizando con TF-IDF...")

vectorizador = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 3)
)

X_entrenar_tfidf = vectorizador.fit_transform(X_entrenar)
X_examen_tfidf = vectorizador.transform(X_examen)

# Combinar con features manuales
X_entrenar_final = hstack([X_entrenar_tfidf, feat_train])
X_examen_final = hstack([X_examen_tfidf, feat_test])

# =========================
# MODELO (SIN CAMBIOS)
# =========================
print("5. Entrenando el SVM...")

modelo_svm = SVC(kernel='linear', C=1, probability=True, class_weight='balanced')
modelo_svm.fit(X_entrenar_final, y_entrenar)

# =========================
# EVALUACIÓN
# =========================
y_pred_entrenar = modelo_svm.predict(X_entrenar_final)
precision_entrenar = accuracy_score(y_entrenar, y_pred_entrenar)

print(f"Precisión en Entrenamiento: {precision_entrenar * 100:.2f}%")

print("6. Evaluando el modelo...")

y_pred = modelo_svm.predict(X_examen_final)
precision = accuracy_score(y_examen, y_pred)

print(f"\n¡Precisión general: {precision * 100:.2f}%\n")

print("Reporte detallado:")
print(classification_report(y_examen, y_pred))

# MATRIZ DE CONFUSIÓN
print("Matriz de confusión:")
print(confusion_matrix(y_examen, y_pred))

print(f"\nBrecha: {(precision_entrenar - precision) * 100:.2f}%")

# =========================
# ANÁLISIS DE ERRORES
# =========================
print("\n--- ERRORES IMPORTANTES ---")

errores = pd.DataFrame({
    "texto": X_examen,
    "real": y_examen,
    "pred": y_pred
})

errores = errores[errores["real"] != errores["pred"]]

print("\nFalsos negativos (spam detectado como ham):")
print(errores[(errores["real"] == "spam") & (errores["pred"] == "ham")].head(5))

print("\nFalsos positivos (ham detectado como spam):")
print(errores[(errores["real"] == "ham") & (errores["pred"] == "spam")].head(5))

# =========================
# GUARDADO
# =========================
print("7. Guardando...")

carpeta_modelo = "Modelo_NLP"
os.makedirs(carpeta_modelo, exist_ok=True)

with open(f"{carpeta_modelo}/vectorizador.pkl", "wb") as f:
    pickle.dump(vectorizador, f)

with open(f"{carpeta_modelo}/modelo_svm.pkl", "wb") as f:
    pickle.dump(modelo_svm, f)

print("Modelo guardado correctamente.")