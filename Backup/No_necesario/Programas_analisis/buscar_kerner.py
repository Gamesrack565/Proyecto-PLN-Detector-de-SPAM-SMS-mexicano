# =============================================================================
#  comparar_kernels.py
#  Entrena y evalúa el modelo SVM con los 4 kernels disponibles y genera
#  una tabla comparativa de métricas para justificar la elección del kernel.
#
#  Ejecutar DESPUÉS de tener el dataset final listo.
#  Requiere: pandas, scikit-learn, spacy, scipy, matplotlib, tabulate
#    pip install tabulate
# =============================================================================

import pandas as pd
import spacy
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack

print("Cargando spaCy...")
nlp = spacy.load("es_core_news_sm")

# ─────────────────────────────────────────────────────────────────────────────
# Preprocesamiento (igual que tu script principal)
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
# Carga y preparación del dataset
# ─────────────────────────────────────────────────────────────────────────────
print("Leyendo dataset...")
ruta_csv = "datos/dataset_spam_modelo.csv"   # <-- ajusta si es necesario
df = pd.read_csv(ruta_csv)
df = df.dropna(subset=['mensaje', 'etiqueta'])
df = df.drop_duplicates(subset=['mensaje'])

print(f"Dataset: {len(df)} mensajes | {df['etiqueta'].value_counts().to_dict()}")

# Split primero, limpiar después
X_raw, y = df['mensaje'], df['etiqueta']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

print("Limpiando textos...")
X_train_clean = X_train_raw.apply(limpiar_texto)
X_test_clean  = X_test_raw.apply(limpiar_texto)

# Eliminar vacíos
mask_tr = X_train_clean.str.strip() != ""
X_train_clean, y_train, X_train_raw = X_train_clean[mask_tr], y_train[mask_tr], X_train_raw[mask_tr]
mask_te = X_test_clean.str.strip() != ""
X_test_clean, y_test, X_test_raw = X_test_clean[mask_te], y_test[mask_te], X_test_raw[mask_te]

# Features manuales
feat_train = np.array([extraer_features(t) for t in X_train_raw])
feat_test  = np.array([extraer_features(t) for t in X_test_raw])

# TF-IDF
print("Vectorizando...")
vec = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9, ngram_range=(1, 3))
X_tr = hstack([vec.fit_transform(X_train_clean), feat_train])
X_te = hstack([vec.transform(X_test_clean),  feat_test])

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de los 4 kernels a comparar
# ─────────────────────────────────────────────────────────────────────────────
kernels = [
    {
        "nombre": "linear",
        "config": {"kernel": "linear", "C": 1, "probability": True, "class_weight": "balanced"},
        "descripcion": "Hiperplano recto. Ideal para NLP con alta dimensionalidad.",
        "ventaja": "Rápido, interpretable, excelente en texto",
        "desventaja": "No captura relaciones no lineales",
    },
    {
        "nombre": "rbf",
        "config": {"kernel": "rbf", "C": 1, "gamma": "scale", "probability": True, "class_weight": "balanced"},
        "descripcion": "Fronteras curvas tipo burbuja (Gaussiano).",
        "ventaja": "Potente en datos numéricos tabulares",
        "desventaja": "Lento y propenso a overfitting en texto",
    },
    {
        "nombre": "poly (grado 3)",
        "config": {"kernel": "poly", "degree": 3, "C": 1, "probability": True, "class_weight": "balanced"},
        "descripcion": "Fronteras polinómicas curvas.",
        "ventaja": "Captura interacciones entre características",
        "desventaja": "Muy lento en alta dimensionalidad",
    },
    {
        "nombre": "sigmoid",
        "config": {"kernel": "sigmoid", "C": 1, "probability": True, "class_weight": "balanced"},
        "descripcion": "Imita una neurona individual.",
        "ventaja": "Puente histórico entre SVM y redes neuronales",
        "desventaja": "Rara vez supera a linear o rbf en la práctica",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento y evaluación de cada kernel
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("COMPARATIVA DE KERNELS SVM")
print("=" * 65)

resultados = []

for k in kernels:
    print(f"\nEntrenando kernel={k['nombre']}...")
    t0 = time.time()
    modelo = SVC(**k["config"])
    modelo.fit(X_tr, y_train)
    t_entrena = time.time() - t0

    y_pred_tr = modelo.predict(X_tr)
    y_pred_te = modelo.predict(X_te)

    acc_tr  = accuracy_score(y_train, y_pred_tr) * 100
    acc_te  = accuracy_score(y_test,  y_pred_te) * 100
    f1_te   = f1_score(y_test, y_pred_te, average='macro') * 100
    prec_te = precision_score(y_test, y_pred_te, average='macro') * 100
    rec_te  = recall_score(y_test, y_pred_te, average='macro') * 100
    brecha  = acc_tr - acc_te
    cm      = confusion_matrix(y_test, y_pred_te)
    fp      = cm[0][1]  # ham clasificado como spam
    fn      = cm[1][0]  # spam clasificado como ham

    resultados.append({
        "Kernel":         k["nombre"],
        "Acc. Train %":   round(acc_tr, 2),
        "Acc. Test %":    round(acc_te, 2),
        "Brecha %":       round(brecha, 2),
        "F1 macro %":     round(f1_te, 2),
        "Precisión %":    round(prec_te, 2),
        "Recall %":       round(rec_te, 2),
        "Falsos +":       fp,
        "Falsos -":       fn,
        "Tiempo (s)":     round(t_entrena, 1),
        "descripcion":    k["descripcion"],
        "ventaja":        k["ventaja"],
        "desventaja":     k["desventaja"],
    })

    print(f"  Train={acc_tr:.2f}% | Test={acc_te:.2f}% | Brecha={brecha:.2f}% | "
          f"F1={f1_te:.2f}% | FP={fp} | FN={fn} | Tiempo={t_entrena:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
# Tabla comparativa en consola
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("TABLA RESUMEN")
print("=" * 65)
df_res = pd.DataFrame(resultados)
cols = ["Kernel", "Acc. Train %", "Acc. Test %", "Brecha %",
        "F1 macro %", "Falsos +", "Falsos -", "Tiempo (s)"]
print(df_res[cols].to_string(index=False))

# Determinar el mejor kernel
mejor = df_res.loc[df_res["F1 macro %"].idxmax(), "Kernel"]
print(f"\n✓ Mejor kernel según F1 macro: '{mejor}'")

# ─────────────────────────────────────────────────────────────────────────────
# Gráfica comparativa
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Comparativa de kernels SVM — Detección de spam", fontsize=13, y=1.02)

nombres   = df_res["Kernel"].tolist()
colores   = ["#2ecc71" if n == mejor else "#95a5a6" for n in nombres]

# Gráfica 1 — Accuracy train vs test
ax1 = axes[0]
x = np.arange(len(nombres))
w = 0.35
ax1.bar(x - w/2, df_res["Acc. Train %"], w, label="Train", color="#3498db", alpha=0.85)
ax1.bar(x + w/2, df_res["Acc. Test %"],  w, label="Test",  color=colores,   alpha=0.85)
ax1.set_xticks(x)
ax1.set_xticklabels(nombres, rotation=15, ha="right", fontsize=9)
ax1.set_ylabel("Accuracy %")
ax1.set_title("Accuracy train vs test")
ax1.legend(fontsize=8)
ax1.set_ylim(0, 110)
for bar, v in zip(ax1.patches[len(nombres):], df_res["Acc. Test %"]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f}", ha="center", va="bottom", fontsize=8)

# Gráfica 2 — F1 macro
ax2 = axes[1]
bars = ax2.bar(nombres, df_res["F1 macro %"], color=colores, alpha=0.85)
ax2.set_ylabel("F1 macro %")
ax2.set_title("F1-score macro (test)")
ax2.set_xticklabels(nombres, rotation=15, ha="right", fontsize=9)
ax2.set_ylim(0, 110)
for bar, v in zip(bars, df_res["F1 macro %"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{v:.1f}", ha="center", va="bottom", fontsize=8)

# Gráfica 3 — Tiempo de entrenamiento
ax3 = axes[2]
bars3 = ax3.bar(nombres, df_res["Tiempo (s)"], color="#e67e22", alpha=0.8)
ax3.set_ylabel("Segundos")
ax3.set_title("Tiempo de entrenamiento")
ax3.set_xticklabels(nombres, rotation=15, ha="right", fontsize=9)
for bar, v in zip(bars3, df_res["Tiempo (s)"]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f"{v:.1f}s", ha="center", va="bottom", fontsize=8)

mejor_patch = mpatches.Patch(color="#2ecc71", alpha=0.85, label=f"Mejor: {mejor}")
fig.legend(handles=[mejor_patch], loc="upper right", fontsize=9)

plt.tight_layout()
plt.savefig("comparativa_kernels.png", dpi=150, bbox_inches="tight")
print("\nGráfica guardada: comparativa_kernels.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# Conclusión automática
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ANÁLISIS Y CONCLUSIÓN")
print("=" * 65)
for _, r in df_res.iterrows():
    marca = "★ RECOMENDADO" if r["Kernel"] == mejor else ""
    print(f"\n{r['Kernel']} {marca}")
    print(f"  {r['descripcion']}")
    print(f"  Ventaja:    {r['ventaja']}")
    print(f"  Desventaja: {r['desventaja']}")
    print(f"  F1={r['F1 macro %']:.1f}% | Brecha={r['Brecha %']:.1f}% | "
          f"FP={r['Falsos +']} | FN={r['Falsos -']} | {r['Tiempo (s)']}s")