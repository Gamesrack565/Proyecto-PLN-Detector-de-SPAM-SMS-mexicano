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
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

print("1. Cargando el modelo de lenguaje de Spacy (español)...")
nlp = spacy.load("es_core_news_sm")

#Correccion de limpieza
def limpiar_texto(texto):
    texto = str(texto).lower()

    #URLs
    texto = re.sub(r'(https?://\S+|www\.\S+)', ' tokenurl ', texto)

    #Correos
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)

    #Números por rangos (mejor que uno solo)
    texto = re.sub(r"\b\d{5,}\b", " tokennumero_grande ", texto)
    texto = re.sub(r"\b\d{2,4}\b", " tokennumero_medio ", texto)
    texto = re.sub(r"\b\d\b", " tokennumero_pequeno ", texto)

    #Menciones
    texto = re.sub(r"[@#]\w+", " ", texto)

    #Mantener solo letras, números y signos útiles
    texto = re.sub(r"[^a-záéíóúüñ0-9!$ ]", " ", texto)

    doc = nlp(texto)

    tokens = []
    for token in doc:
        if not token.is_space:
            tokens.append(token.text)

    return " ".join(tokens)


#Extraccion de features manuales
def extraer_features(texto):
    texto = str(texto)

    longitud = len(texto)
    num_palabras = len(texto.split())
    num_numeros = len(re.findall(r'\d', texto))
    num_urls = texto.count("tokenurl")
    num_exclamaciones = texto.count("!")
    num_dolar = texto.count("$")

    return [longitud, num_palabras, num_numeros, num_urls, num_exclamaciones, num_dolar]

#Proceso de lectura del snetro de datos
print("2. Leyendo el dataset...")
ruta_csv = "datos/spam_espanol.csv"
df = pd.read_csv(ruta_csv)
#Se obtienen etiquetas
df = df.dropna(subset=['mensaje', 'etiqueta'])
#Proceso de limpieza
print(f"3. Limpiando {len(df)} mensajes...")
df['mensaje_limpio'] = df['mensaje'].apply(limpiar_texto)

#Eliminar vacíos
df = df[df['mensaje_limpio'].str.strip() != ""]

#CHEQUEO DE DUPLICADOS
print("=== ANÁLISIS DE DUPLICADOS ===")
#Obtenemos el número total de mensajes
print(f"Mensajes totales: {len(df)}")
#Contamos duplicados exactos
duplicados = df.duplicated(subset=['mensaje_limpio']).sum()
#Obtenemos los numeros de duplicados
print(f"Duplicados exactos: {duplicados}")

#Eliminar duplicados exactos
df = df.drop_duplicates(subset=['mensaje_limpio'])

#PROCESO DE DUPLICADOS SEMÁNTICOS
print("Eliminando duplicados semánticos (esto puede tardar)...")
#Usamos TF-IDF para obtener representaciones vectoriales de los mensajes limpios
#Vector_temp se usa solo para este proceso, no afecta al vectorizador final
vector_temp = TfidfVectorizer(max_features=3000)
#Obtenemos la matriz de similitud coseno entre los mensajes
X_temp = vector_temp.fit_transform(df['mensaje_limpio'])
sim_matrix = cosine_similarity(X_temp)

los_que_elimin = set()
umbral_tolerancia = 0.90

for i in range(sim_matrix.shape[0]):
    for j in range(i + 1, sim_matrix.shape[1]):
        if sim_matrix[i, j] > umbral_tolerancia:
            los_que_elimin.add(j)

print(f"\n--- REPORTE DE DUPLICADOS SEMÁNTICOS ---")
print(f"Se detectaron {len(los_que_elimin)} mensajes casi idénticos.")

if len(los_que_elimin) > 0:
    print("Ejemplos de mensajes que serán eliminados:")
    #Tomamos solo los primeros 5 índices para no inundar la pantalla
    indices_ejemplo = list(los_que_elimin)[:10] 
    
    #Extraemos el texto original usando iloc (ya que los índices son posicionales)
    ejemplos_texto = df.iloc[indices_ejemplo]['mensaje'].values
    
    for idx, texto in enumerate(ejemplos_texto, 1):
        print(f"  {idx}. {texto}")
print("------------------------------------------\n")

df = df.drop(df.index[list(los_que_elimin)])

print(f"Mensajes después de eliminar duplicados semánticos: {len(df)}")

#PREPARACION
X = df['mensaje_limpio']
y = df['etiqueta']

#Features adicionales
features_extra = np.array([extraer_features(t) for t in df['mensaje']])

#Division en entrenamiento y examen
X_entrenar, X_examen, y_entrenar, y_examen, feat_train, feat_test = train_test_split(
    X, y, features_extra, test_size=0.2, stratify=y
)

print("\n--- Muestras de ENTRENAMIENTO ---")
print(X_entrenar.head(3))

print("\n--- Muestras de EXAMEN ---")
print(X_examen.head(3))

#TF - IDF actualizado
print("4. Vectorizando con TF-IDF...")
#Max_features = 5000 para capturar más vocabulario, min_df=2 para eliminar palabras muy raras, max_df=0.9 para eliminar palabras muy comunes, ngram_range=(1,3) para capturar un poco de contexto
vectorizador = TfidfVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 3)
)

X_entrenar_tfidf = vectorizador.fit_transform(X_entrenar)
X_examen_tfidf = vectorizador.transform(X_examen)

#Combinar con features manuales
X_entrenar_final = hstack([X_entrenar_tfidf, feat_train])
X_examen_final = hstack([X_examen_tfidf, feat_test])

#PREPARACION DE MODELO
print("5. Entrenando el SVM...")
#Usamos un SVM con kernel lineal, C=1 para un buen equilibrio entre margen y errores, probability=True para obtener probabilidades de predicción, class_weight='balanced' para manejar el desbalance de clases
#class_weight= 'balanced'
modelo_svm = SVC(kernel='linear', C=1, probability=True, class_weight='balanced')
modelo_svm.fit(X_entrenar_final, y_entrenar)

#EValuacion en entrenamiento para detectar overfitting
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

#Analisis de errores
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


print("ANALISIS------00")
# ==========================================
# 1. VISUALIZACIÓN DE CLASES (REDUCCIÓN 2D)
# ==========================================
print("\nGenerando visualización 2D de las clases...")

# Usamos TruncatedSVD en lugar de PCA para no colapsar la memoria con la matriz TF-IDF
reductor = TruncatedSVD(n_components=2, random_state=42)
X_2d = reductor.fit_transform(X_entrenar_final)

plt.figure(figsize=(10, 6))
# Dibujamos los puntos: azul para ham, rojo para spam
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
# plt.savefig("grafico_clases.png", dpi=300) # Descomenta esto si quieres guardar la imagen
plt.show()

# ==========================================
# 2. VOCABULARIO Y FRECUENCIAS
# ==========================================
print("\n--- ANÁLISIS DEL VOCABULARIO FINAL ---")

# Extraemos las palabras que el TF-IDF seleccionó
vocabulario = vectorizador.get_feature_names_out()

# Sumamos los pesos de cada palabra en todo el set de entrenamiento
# Nota: Al usar TF-IDF, esto nos da la "Importancia Total" de la palabra, no solo el conteo bruto
importancias_tfidf = X_entrenar_tfidf.sum(axis=0).A1 

# Creamos un DataFrame para ordenarlo y visualizarlo bien
df_vocabulario = pd.DataFrame({
    'Termino': vocabulario, 
    'Peso_TFIDF_Total': importancias_tfidf
})

# Ordenamos de mayor a menor importancia
df_vocabulario = df_vocabulario.sort_values(by='Peso_TFIDF_Total', ascending=False)

print(f"Tamaño total del vocabulario final: {len(vocabulario)} términos (palabras y n-gramas).")

print("\nTop 15 términos más importantes/frecuentes en el modelo:")
print(df_vocabulario.head(15).to_string(index=False))

print("\nLos 5 términos menos frecuentes (a punto de ser eliminados por min_df):")
print(df_vocabulario.tail(5).to_string(index=False))
print("------------------------------------------\n")


#Guardado
print("7. Guardando...")

carpeta_modelo = "Modelo_NLP"
os.makedirs(carpeta_modelo, exist_ok=True)

with open(f"{carpeta_modelo}/vectorizador.pkl", "wb") as f:
    pickle.dump(vectorizador, f)

with open(f"{carpeta_modelo}/modelo_svm.pkl", "wb") as f:
    pickle.dump(modelo_svm, f)

print("Modelo guardado correctamente.")