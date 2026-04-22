import pandas as pd
import spacy
import re
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import hstack

print("1. Cargando el modelo de lenguaje de Spacy (español)...")
nlp = spacy.load("es_core_news_sm")

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
    # Agregamos 'and not token.is_stop' para filtrar las palabras vacías
    tokens = [token.text for token in doc if not token.is_space and not token.is_stop]
    return " ".join(tokens)

def extraer_features(texto):
    texto = str(texto)
    longitud = len(texto)
    num_palabras = len(texto.split())
    num_numeros = len(re.findall(r'\d', texto))
    num_urls = texto.count("tokenurl")
    num_exclamaciones = texto.count("!")
    num_dolar = texto.count("$")
    return [longitud, num_palabras, num_numeros, num_urls, num_exclamaciones, num_dolar]

# ──────────────────────────────────────────────────────────────────────────────
# CORRECCIÓN 1: Eliminar duplicados ANTES de todo, usando el mensaje original
# ──────────────────────────────────────────────────────────────────────────────
print("2. Leyendo el dataset...")
ruta_csv = "datos/spam_espanol.csv"       # <-- ajusta la ruta según tu proyecto
df = pd.read_csv(ruta_csv)
df = df.dropna(subset=['mensaje', 'etiqueta'])

print(f"Mensajes totales: {len(df)}")
duplicados_exactos = df.duplicated(subset=['mensaje']).sum()
print(f"Duplicados exactos: {duplicados_exactos}")

# Eliminamos duplicados exactos por mensaje original (antes de limpiar)
df = df.drop_duplicates(subset=['mensaje'])
print(f"Mensajes después de eliminar duplicados exactos: {len(df)}")
print(df['etiqueta'].value_counts())

# ──────────────────────────────────────────────────────────────────────────────
# CORRECCIÓN 2: Hacer el train/test split PRIMERO, antes de cualquier
# transformación que use todo el dataset (limpieza, TF-IDF, similitud coseno).
# Así el conjunto de prueba nunca "contamina" al de entrenamiento.
# ──────────────────────────────────────────────────────────────────────────────
print("\n3. Dividiendo el dataset en entrenamiento y prueba ANTES de procesar...")
X_raw = df['mensaje']
y = df['etiqueta']

X_entrenar_raw, X_examen_raw, y_entrenar, y_examen = train_test_split(
    X_raw, y, test_size=0.2, stratify=y
)

print(f"Entrenamiento: {len(X_entrenar_raw)} mensajes")
print(f"Prueba:        {len(X_examen_raw)} mensajes")

# ──────────────────────────────────────────────────────────────────────────────
# Limpieza: se aplica por separado a cada partición
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n4. Limpiando mensajes de entrenamiento ({len(X_entrenar_raw)})...")
X_entrenar = X_entrenar_raw.apply(limpiar_texto)

print(f"   Limpiando mensajes de prueba ({len(X_examen_raw)})...")
X_examen = X_examen_raw.apply(limpiar_texto)

# Eliminamos vacíos resultantes de la limpieza (borrar también la etiqueta)
mascara_train = X_entrenar.str.strip() != ""
X_entrenar = X_entrenar[mascara_train]
y_entrenar  = y_entrenar[mascara_train]
X_entrenar_raw = X_entrenar_raw[mascara_train]

mascara_test = X_examen.str.strip() != ""
X_examen = X_examen[mascara_test]
y_examen = y_examen[mascara_test]
X_examen_raw = X_examen_raw[mascara_test]

# ──────────────────────────────────────────────────────────────────────────────
# NOTA SOBRE DUPLICADOS SEMÁNTICOS:
# Se elimina el bloque de similitud coseno global que existía en el código
# original porque calculaba la similitud sobre TODOS los datos antes del split,
# lo que causaba fuga de información (data leakage). Si deseas eliminar
# duplicados semánticos del conjunto de entrenamiento, hazlo aquí — SOLO sobre
# X_entrenar — y nunca toques X_examen.
# ──────────────────────────────────────────────────────────────────────────────

# Features manuales (calculadas sobre el mensaje original, no el limpio)
feat_train = np.array([extraer_features(t) for t in X_entrenar_raw])
feat_test  = np.array([extraer_features(t) for t in X_examen_raw])

# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF: fit SOLO en entrenamiento, transform en ambos
# ──────────────────────────────────────────────────────────────────────────────
print("\n5. Vectorizando con TF-IDF...")
vectorizador = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 3)
)

X_entrenar_tfidf = vectorizador.fit_transform(X_entrenar)   # fit solo aquí
X_examen_tfidf   = vectorizador.transform(X_examen)         # solo transform

X_entrenar_final = hstack([X_entrenar_tfidf, feat_train])
X_examen_final   = hstack([X_examen_tfidf,   feat_test])

# ──────────────────────────────────────────────────────────────────────────────
# Entrenamiento del SVM
# ──────────────────────────────────────────────────────────────────────────────
print("\n6. Entrenando el SVM...")
modelo_svm = SVC(kernel='linear', C=1, probability=True, class_weight='balanced')
modelo_svm.fit(X_entrenar_final, y_entrenar)

y_pred_entrenar = modelo_svm.predict(X_entrenar_final)
precision_entrenar = accuracy_score(y_entrenar, y_pred_entrenar)
print(f"Precisión en Entrenamiento: {precision_entrenar * 100:.2f}%")

print("\n7. Evaluando en el conjunto de PRUEBA...")
y_pred = modelo_svm.predict(X_examen_final)
precision = accuracy_score(y_examen, y_pred)

print(f"\n¡Precisión general en prueba: {precision * 100:.2f}%\n")
print("Reporte detallado:")
print(classification_report(y_examen, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_examen, y_pred))

brecha = (precision_entrenar - precision) * 100
print(f"\nBrecha entrenamiento/prueba: {brecha:.2f}%")
if brecha > 10:
    print("⚠  Brecha alta — posible sobreajuste.")
else:
    print("✓  Brecha razonable.")

# ──────────────────────────────────────────────────────────────────────────────
# Análisis de errores
# ──────────────────────────────────────────────────────────────────────────────
print("\n--- ERRORES IMPORTANTES ---")
errores = pd.DataFrame({
    "texto": X_examen,
    "real": y_examen,
    "pred": y_pred
})
errores = errores[errores["real"] != errores["pred"]]

print("\nFalsos negativos (spam detectado como ham):")
print(errores[(errores["real"] == "spam") & (errores["pred"] == "ham")].head(5).to_string())

print("\nFalsos positivos (ham detectado como spam):")
print(errores[(errores["real"] == "ham") & (errores["pred"] == "spam")].head(5).to_string())

# ──────────────────────────────────────────────────────────────────────────────
# Visualizaciones
# ──────────────────────────────────────────────────────────────────────────────
print("\nGenerando visualización 2D de las clases...")
reductor = TruncatedSVD(n_components=2, random_state=42)
X_2d = reductor.fit_transform(X_entrenar_final)

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_2d[:, 0],
    y=X_2d[:, 1],
    hue=y_entrenar,
    palette={'ham': '#1f77b4', 'spam': '#d62728'},
    alpha=0.6,
    edgecolor=None
)
plt.title('Visualización del Dataset (Spam vs Ham) en 2D')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Clase')
plt.tight_layout()
plt.show()

print("\n--- ANÁLISIS DEL VOCABULARIO FINAL ---")
vocabulario = vectorizador.get_feature_names_out()
importancias_tfidf = X_entrenar_tfidf.sum(axis=0).A1
df_vocabulario = pd.DataFrame({'Termino': vocabulario, 'Peso_TFIDF_Total': importancias_tfidf})
df_vocabulario = df_vocabulario.sort_values(by='Peso_TFIDF_Total', ascending=False)

print(f"Tamaño total del vocabulario: {len(vocabulario)} términos.")
print("\nTop 15 términos más importantes:")
print(df_vocabulario.head(15).to_string(index=False))

# ──────────────────────────────────────────────────────────────────────────────
# Guardado
# ──────────────────────────────────────────────────────────────────────────────
print("\n8. Guardando modelo...")
carpeta_modelo = "Modelo_NLP"
os.makedirs(carpeta_modelo, exist_ok=True)

with open(f"{carpeta_modelo}/vectorizador.pkl", "wb") as f:
    pickle.dump(vectorizador, f)

with open(f"{carpeta_modelo}/modelo_svm.pkl", "wb") as f:
    pickle.dump(modelo_svm, f)

print("Modelo guardado correctamente.")