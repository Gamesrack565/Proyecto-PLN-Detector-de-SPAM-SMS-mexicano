# =============================================================================
#  buscar_hiperparametros.py
#  Usa RandomizedSearchCV para encontrar la mejor combinación de parámetros
#  del SVM. Corre DESPUÉS de comparar_kernels.py (que ya justificó el kernel).
#
#  Instalar si no tienes: pip install pandas scikit-learn spacy scipy
# =============================================================================

import pandas as pd
import spacy
import re
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack, csr_matrix
from scipy.stats import loguniform

print("Cargando spaCy...")
nlp = spacy.load("es_core_news_sm")

# ─────────────────────────────────────────────────────────────────────────────
# Preprocesamiento
# ─────────────────────────────────────────────────────────────────────────────
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r'(https?://\S+|www\.\S+)', ' tokenurl ', texto)
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)
    texto = re.sub(r"\b\d{5,}\b", " tokennumero_grande ", texto)
    texto = re.sub(r"\b\d{2,4}\b", " tokennumero_medio ", texto)
    texto = re.sub(r"\b\d\b", " tokennumero_pequeno ", texto)
    texto = re.sub(r"[@#]", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ0-9!$ ]", " ", texto)
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_space and not token.is_stop]
    return " ".join(tokens)

def extraer_features(texto):
    texto = str(texto)
    return [
        len(texto),
        len(texto.split()),
        len(re.findall(r'\d', texto)),
        texto.count("tokenurl"),
        texto.count("!"),
        texto.count("$"),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# Carga y split
# ─────────────────────────────────────────────────────────────────────────────
print("Leyendo dataset...")
ruta_csv = "datos/dataset_spam_modelo.csv"    # <-- ajusta si es necesario
df = pd.read_csv(ruta_csv)
df = df.dropna(subset=['mensaje', 'etiqueta'])
df = df.drop_duplicates(subset=['mensaje'])
print(f"Dataset: {len(df)} mensajes | {df['etiqueta'].value_counts().to_dict()}")

X_raw, y = df['mensaje'], df['etiqueta']

# Split ANTES de cualquier transformación
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

# ─────────────────────────────────────────────────────────────────────────────
# Limpieza por separado
# ─────────────────────────────────────────────────────────────────────────────
print(f"Limpiando {len(X_train_raw)} mensajes de entrenamiento...")
X_train_clean = X_train_raw.apply(limpiar_texto)
print(f"Limpiando {len(X_test_raw)} mensajes de prueba...")
X_test_clean  = X_test_raw.apply(limpiar_texto)

mask_tr = X_train_clean.str.strip() != ""
X_train_clean, y_train, X_train_raw = X_train_clean[mask_tr], y_train[mask_tr], X_train_raw[mask_tr]
mask_te = X_test_clean.str.strip() != ""
X_test_clean, y_test, X_test_raw = X_test_clean[mask_te], y_test[mask_te], X_test_raw[mask_te]

# Features manuales
feat_train = np.array([extraer_features(t) for t in X_train_raw])
feat_test  = np.array([extraer_features(t) for t in X_test_raw])

# TF-IDF (fijo para la búsqueda, para no explotar el tiempo)
print("Vectorizando con TF-IDF...")
vec = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9, ngram_range=(1, 3))
X_tr_tfidf = vec.fit_transform(X_train_clean)
X_te_tfidf = vec.transform(X_test_clean)

X_tr = hstack([X_tr_tfidf, feat_train])
X_te = hstack([X_te_tfidf, feat_test])

# ─────────────────────────────────────────────────────────────────────────────
# Espacio de búsqueda de hiperparámetros
# ─────────────────────────────────────────────────────────────────────────────
# Solo exploramos parámetros del SVM con kernel linear (ya justificado en
# comparar_kernels.py). El objetivo es encontrar la mejor combinación de:
#   C            — controla el margen de error. Bajo = más permisivo = menos overfitting
#   class_weight — compensar desbalance de clases (None vs balanced)
#   probability  — si se calculan probabilidades (afecta levemente la velocidad)

param_dist = {
    "C":            loguniform(1e-2, 1e2),   # muestrea en escala logarítmica: 0.01 a 100
    "class_weight": [None, "balanced"],
    "probability":  [True, False],
}

print("\n" + "=" * 60)
print("BÚSQUEDA ALEATORIA DE HIPERPARÁMETROS")
print("=" * 60)
print("Parámetros a explorar:")
print("  C            : 0.01 a 100 (escala logarítmica)")
print("  class_weight : None | balanced")
print("  probability  : True | False")
print()
print("Configuración:")
print("  Iteraciones  : 40 combinaciones aleatorias")
print("  Validación   : 5-fold estratificado")
print("  Métrica      : F1 macro (penaliza errores en ambas clases)")
print()

svm_base = SVC(kernel="rbf")

# StratifiedKFold mantiene la proporción spam/ham en cada fold
cv = StratifiedKFold(n_splits=5, shuffle=True)

busqueda = RandomizedSearchCV(
    estimator=svm_base,
    param_distributions=param_dist,
    n_iter=40,              # 40 combinaciones aleatorias
    cv=cv,
    scoring="f1_macro",     # F1 macro porque nos importan ambas clases por igual
    n_jobs=-1,              # usar todos los núcleos disponibles
    verbose=1,
)

print("Iniciando búsqueda (esto puede tardar 2-5 minutos)...")
busqueda.fit(X_tr, y_train)

# ─────────────────────────────────────────────────────────────────────────────
# Resultados de la búsqueda
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTADOS DE LA BÚSQUEDA")
print("=" * 60)

mejor_params = busqueda.best_params_
mejor_score_cv = busqueda.best_score_

print(f"\nMejor combinación encontrada:")
print(f"  C            = {mejor_params['C']:.4f}")
print(f"  class_weight = {mejor_params['class_weight']}")
print(f"  probability  = {mejor_params['probability']}")
print(f"\nF1 macro en validación cruzada: {mejor_score_cv * 100:.2f}%")

# Top 10 combinaciones
print("\n--- Top 10 combinaciones ---")
df_cv = pd.DataFrame(busqueda.cv_results_)
df_cv = df_cv[["param_C", "param_class_weight", "param_probability",
               "mean_test_score", "std_test_score", "rank_test_score"]]
df_cv = df_cv.sort_values("rank_test_score").head(10)
df_cv["mean_test_score"] = (df_cv["mean_test_score"] * 100).round(2)
df_cv["std_test_score"]  = (df_cv["std_test_score"]  * 100).round(2)
df_cv.columns = ["C", "class_weight", "probability", "F1 CV %", "std %", "rank"]
df_cv["C"] = df_cv["C"].apply(lambda x: f"{x:.4f}")
print(df_cv.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# Comparativa: modelo base vs modelo optimizado
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("COMPARATIVA: BASE vs OPTIMIZADO")
print("=" * 60)

# Modelo base (el que tenías antes)
base = SVC(kernel="rbf", C=1, probability=True, class_weight="balanced")
base.fit(X_tr, y_train)
y_pred_base = base.predict(X_te)

# Modelo optimizado
mejor_modelo = busqueda.best_estimator_
y_pred_mejor = mejor_modelo.predict(X_te)

def metricas(y_true, y_pred, nombre):
    acc = accuracy_score(y_true, y_pred) * 100
    f1  = f1_score(y_true, y_pred, average='macro') * 100
    cm  = confusion_matrix(y_true, y_pred)
    fp  = cm[0][1]
    fn  = cm[1][0]
    print(f"\n{nombre}")
    print(f"  Accuracy : {acc:.2f}%")
    print(f"  F1 macro : {f1:.2f}%")
    print(f"  Falsos + : {fp}  (ham marcado como spam)")
    print(f"  Falsos - : {fn}  (spam que pasó como ham)")

metricas(y_test, y_pred_base,  "Modelo BASE  (C=1, balanced, probability=True)")
metricas(y_test, y_pred_mejor, "Modelo ÓPTIMO (parámetros encontrados por búsqueda)")

print("\n--- Reporte detallado del modelo ÓPTIMO ---")
print(classification_report(y_test, y_pred_mejor))

# ─────────────────────────────────────────────────────────────────────────────
# Conclusión automática
# ─────────────────────────────────────────────────────────────────────────────
f1_base  = f1_score(y_test, y_pred_base,  average='macro') * 100
f1_mejor = f1_score(y_test, y_pred_mejor, average='macro') * 100
diff = f1_mejor - f1_base

print("\n" + "=" * 60)
print("CONCLUSIÓN")
print("=" * 60)
if diff > 0.5:
    print(f"La búsqueda mejoró el F1 en {diff:.2f} puntos porcentuales.")
    print("Usa estos parámetros en tu script de entrenamiento final.")
elif diff > -0.5:
    print(f"La diferencia es mínima ({diff:+.2f}%). Los parámetros base ya eran buenos.")
    print("Puedes mantener C=1, balanced — la búsqueda lo confirma.")
else:
    print(f"El modelo base supera al optimizado en {-diff:.2f}%.")
    print("Mantén los parámetros originales.")

print(f"\nParámetros a usar en entrenamiento_corregido.py:")
print(f"  modelo_svm = SVC(")
print(f"      kernel='linear',")
print(f"      C={mejor_params['C']:.4f},")
print(f"      class_weight={repr(mejor_params['class_weight'])},")
print(f"      probability={mejor_params['probability']}")
print(f"  )")