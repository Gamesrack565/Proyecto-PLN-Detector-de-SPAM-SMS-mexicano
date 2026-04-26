# =============================================================================
#  construir_dataset.py
#  Fusiona 3 fuentes de datos para construir un dataset de spam en español
#  con contexto mexicano.
#
#  FUENTES:
#    1. Dataset México (Kaggle)      — ya lo tienes descargado
#    2. Dataset español (Hugging Face) — se descarga automáticamente
#    3. Dataset UCI en inglés        — se descarga y traduce automáticamente
#
#  PREREQUISITOS (instala una sola vez en tu terminal de VS Code):
#    pip install pandas requests deep-translator tqdm datasets
# =============================================================================

import pandas as pd
import requests
import io
import re
import os
import time
from tqdm import tqdm
from deep_translator import GoogleTranslator

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta esta ruta a donde tienes tu dataset de Kaggle México
# ─────────────────────────────────────────────────────────────────────────────
RUTA_DATASET_MEXICO = "dataset.csv"   # <-- pon aquí la ruta de tu archivo

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — Cargar dataset México (Kaggle)
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PASO 1: Cargando dataset México (Kaggle)...")
print("=" * 60)

df_mexico = pd.read_csv(RUTA_DATASET_MEXICO)

# Normalizar columnas a: mensaje, etiqueta
df_mexico = df_mexico.rename(columns={
    "texto": "mensaje",
    "text":  "mensaje",
    "sms":   "mensaje",
})
df_mexico = df_mexico[["mensaje", "etiqueta"]].copy()
df_mexico["etiqueta"] = df_mexico["etiqueta"].str.strip().str.lower()
df_mexico["fuente"] = "kaggle_mexico"

print(f"  Cargados: {len(df_mexico)} mensajes")
print(f"  {df_mexico['etiqueta'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — Descargar dataset español (Hugging Face)
#           Descargamos los CSV directamente porque el viewer tiene un bug
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PASO 2: Descargando dataset español (Hugging Face)...")
print("=" * 60)

HF_URLS = [
    "https://huggingface.co/datasets/softecapps/spam_ham_spanish/resolve/main/train.csv",
    "https://huggingface.co/datasets/softecapps/spam_ham_spanish/resolve/main/test.csv",
]

partes_hf = []
for url in HF_URLS:
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        df_part = pd.read_csv(io.StringIO(resp.text))
        # El dataset HF tiene un espacio extra en " tipo" en el test split
        df_part.columns = df_part.columns.str.strip()
        partes_hf.append(df_part)
        print(f"  ✓ Descargado: {url.split('/')[-1]} — {len(df_part)} filas")
    except Exception as e:
        print(f"  ✗ No se pudo descargar {url.split('/')[-1]}: {e}")

if partes_hf:
    df_hf = pd.concat(partes_hf, ignore_index=True)
    # Normalizar columnas (puede llamarse 'mensaje'/'tipo' o 'Mensaje'/'Etiqueta')
    df_hf.columns = df_hf.columns.str.strip().str.lower()
    col_texto = next((c for c in df_hf.columns if c in ["mensaje", "text", "sms", "message"]), None)
    col_label = next((c for c in df_hf.columns if c in ["tipo", "etiqueta", "label"]), None)

    if col_texto and col_label:
        df_hf = df_hf[[col_texto, col_label]].rename(
            columns={col_texto: "mensaje", col_label: "etiqueta"}
        )
        df_hf["etiqueta"] = df_hf["etiqueta"].str.strip().str.lower()
        # Normalizar etiquetas al formato spam/ham
        df_hf["etiqueta"] = df_hf["etiqueta"].replace({"legítimo": "ham", "legitimo": "ham"})
        df_hf["fuente"] = "huggingface_es"
        print(f"  Total HF: {len(df_hf)} mensajes — {df_hf['etiqueta'].value_counts().to_dict()}")
    else:
        print(f"  ✗ Columnas inesperadas: {df_hf.columns.tolist()} — se omite esta fuente")
        df_hf = pd.DataFrame(columns=["mensaje", "etiqueta", "fuente"])
else:
    print("  ✗ No se pudo descargar el dataset de HuggingFace — se continúa sin él")
    df_hf = pd.DataFrame(columns=["mensaje", "etiqueta", "fuente"])


# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — Descargar UCI en inglés y traducir al español
#           Solo tomamos una muestra balanceada para no inflar el dataset
#           con los mismos mensajes que ya tenías traducidos antes
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PASO 3: Descargando y traduciendo dataset UCI (inglés)...")
print("=" * 60)

UCI_URL = "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

try:
    resp = requests.get(UCI_URL, timeout=30)
    resp.raise_for_status()
    df_uci_raw = pd.read_csv(
        io.StringIO(resp.text),
        encoding="latin-1",
        usecols=[0, 1],
        names=["etiqueta", "mensaje"],
        header=0,
    )
    df_uci_raw["etiqueta"] = df_uci_raw["etiqueta"].str.strip().str.lower()
    print(f"  UCI descargado: {len(df_uci_raw)} mensajes")
    print(f"  {df_uci_raw['etiqueta'].value_counts().to_dict()}")

    # Tomamos SOLO mensajes de smishing/phishing más ambiguos:
    # spam que contenga URLs, números de teléfono o palabras de alerta
    patron_ambiguo = r'(http|www|call|claim|free|win|prize|urgent|verify|account|bank)'
    df_spam_uci = df_uci_raw[
        (df_uci_raw["etiqueta"] == "spam") &
        (df_uci_raw["mensaje"].str.contains(patron_ambiguo, case=False, regex=True))
    ].sample(n=min(200, len(df_uci_raw[df_uci_raw["etiqueta"]=="spam"])), random_state=42)

    # Ham: mensajes cortos y conversacionales (más difíciles de clasificar)
    df_ham_uci = df_uci_raw[
        (df_uci_raw["etiqueta"] == "ham") &
        (df_uci_raw["mensaje"].str.len() < 120)
    ].sample(n=min(200, len(df_uci_raw[df_uci_raw["etiqueta"]=="ham"])), random_state=42)

    df_uci_muestra = pd.concat([df_spam_uci, df_ham_uci], ignore_index=True)
    print(f"  Muestra seleccionada para traducir: {len(df_uci_muestra)} mensajes")

    # Traducción con GoogleTranslator (sin API key, límite 5000 chars por llamada)
    print("  Traduciendo al español (esto tarda ~3-5 minutos, ten paciencia)...")
    translator = GoogleTranslator(source="en", target="es")

    def traducir_seguro(texto):
        """Traduce un texto, con reintento en caso de error."""
        try:
            # Google Translate tiene límite de 5000 chars
            if len(texto) > 4500:
                texto = texto[:4500]
            resultado = translator.translate(texto)
            time.sleep(0.15)   # pequeña pausa para no saturar la API gratuita
            return resultado if resultado else texto
        except Exception:
            time.sleep(1)
            try:
                return translator.translate(texto[:2000])
            except Exception:
                return texto   # si falla dos veces, dejamos el original

    traducciones = []
    for msg in tqdm(df_uci_muestra["mensaje"], desc="  Traduciendo", unit="msg"):
        traducciones.append(traducir_seguro(msg))

    df_uci_traducido = df_uci_muestra.copy()
    df_uci_traducido["mensaje"] = traducciones
    df_uci_traducido["fuente"] = "uci_traducido"
    print(f"  ✓ Traducción completada: {len(df_uci_traducido)} mensajes")

except Exception as e:
    print(f"  ✗ Error descargando UCI: {e}")
    df_uci_traducido = pd.DataFrame(columns=["mensaje", "etiqueta", "fuente"])


# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — Fusión, limpieza y deduplicación
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PASO 4: Fusionando y limpiando...")
print("=" * 60)

df_final = pd.concat(
    [df_mexico, df_hf, df_uci_traducido],
    ignore_index=True
)[["mensaje", "etiqueta", "fuente"]]

print(f"  Total antes de limpiar: {len(df_final)}")

# Eliminar nulos y vacíos
df_final = df_final.dropna(subset=["mensaje", "etiqueta"])
df_final = df_final[df_final["mensaje"].str.strip() != ""]

# Normalizar etiquetas (por si hay variantes)
df_final["etiqueta"] = df_final["etiqueta"].str.strip().str.lower()
etiquetas_validas = {"spam", "ham"}
df_final = df_final[df_final["etiqueta"].isin(etiquetas_validas)]

# Eliminar duplicados exactos
antes = len(df_final)
df_final = df_final.drop_duplicates(subset=["mensaje"])
print(f"  Duplicados exactos eliminados: {antes - len(df_final)}")

print(f"\n  Total final: {len(df_final)} mensajes")
print(f"  Por etiqueta:\n{df_final['etiqueta'].value_counts().to_string()}")
print(f"\n  Por fuente:\n{df_final['fuente'].value_counts().to_string()}")


# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — Guardar
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PASO 5: Guardando...")
print("=" * 60)

SALIDA = "dataset_spam_final.csv"
df_final.to_csv(SALIDA, index=False, encoding="utf-8-sig")
print(f"  ✓ Guardado en: {SALIDA}")

# También guarda una versión solo con las columnas que espera tu modelo
df_modelo = df_final[["mensaje", "etiqueta"]].copy()
df_modelo.columns = ["mensaje", "etiqueta"]   # ya coincide con tu script de entrenamiento
df_modelo.to_csv("dataset_spam_modelo.csv", index=False, encoding="utf-8-sig")
print(f"  ✓ Versión para el modelo: dataset_spam_modelo.csv")

print("\n¡Listo! Ahora usa 'dataset_spam_modelo.csv' en tu script de entrenamiento.")
print("Recuerda cambiar la ruta en tu script:")
print('  ruta_csv = "dataset_spam_modelo.csv"')