#Librerías

#pandas: Manipulación y análisis de datos. Se usa principalmente para cargar tu dataset (CSV) y manejarlo como una tabla (DataFrame).
import pandas as pd

#spacy: Librería de Procesamiento de Lenguaje Natural (NLP). Se usa para entender el texto, separar palabras y eliminar "stop words" (palabras vacías).
import spacy

#re: Expresiones regulares. Te permite buscar, limpiar o reemplazar patrones complejos de texto (ej. encontrar todas las URLs o correos electrónicos).
import re

#numpy: Librería matemática de alto rendimiento. Se utiliza para crear y manipular arreglos (arrays) numéricos a gran velocidad.
import numpy as np

#warnings: Controla los mensajes de advertencia de Python. La segunda línea los "silencia" para que tu consola no se llene de avisos no críticos.
import warnings
warnings.filterwarnings('ignore')

# sklearn.model_selection: Herramientas para evaluar y ajustar cómo aprende tu modelo.
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

#sklearn.feature_extraction.text (TfidfVectorizer): Convierte las palabras de tus mensajes en números, dándole mayor puntuación a las palabras raras e importantes.
from sklearn.feature_extraction.text import TfidfVectorizer

#sklearn.svm (SVC): El algoritmo de Machine Learning. Clasificador de Máquinas de Vectores de Soporte que aprenderá a trazar la frontera entre el Spam y el Ham.
from sklearn.svm import SVC

#sklearn.metrics: Herramientas para calificar qué tan bien lo hizo tu modelo (calificación general, F1, reportes detallados y la matriz para ver dónde se equivocó).
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

#sklearn.pipeline (Pipeline): Te permite empaquetar todo tu proceso (ej. limpieza -> vectorización -> entrenamiento) como si fuera un solo tubo o bloque, evitando errores y fugas de datos.
from sklearn.pipeline import Pipeline

#scipy.sparse: Herramientas matemáticas para manejar matrices "dispersas" (tablas gigantes llenas casi en su totalidad de ceros, típicas del TF-IDF).
from scipy.sparse import hstack, csr_matrix

#scipy.stats (loguniform): Función estadística para generar números aleatorios en una escala logarítmica. Es la mejor forma de alimentar a "RandomizedSearchCV" cuando buscas el valor de 'C' en tu SVM probando desde números enanos (0.001) hasta gigantes (100).
from scipy.stats import loguniform

#Carga del modelo es_core_news
print("Cargando spaCy...")
nlp = spacy.load("es_core_news_sm")

#Proceso de limpieza de texto
def limpiar_texto(texto):
    #Se realiza una limipieza del texto, como:
    #Convierte el texto a minúsculas.
    texto = str(texto).lower()
    #Elimina URLs y reemplaza por un token específico para mantener la información de que había una URL sin conservar la URL en sí.
    texto = re.sub(r'(https?://\S+|www\.\S+)', ' tokenurl ', texto)
    #Elimina direcciones de correo electrónico y reemplaza por un token específico para mantener la información de que había un correo sin conservar el correo en sí.
    texto = re.sub(r"\S+@\S+\.\S+", " tokencorreo ", texto)
    #Reemplaza números grandes, medianos y pequeños por tokens específicos para mantener la información de que había números sin conservar los números en sí.
    texto = re.sub(r"\b\d{5,}\b", " tokennumero_grande ", texto)
    texto = re.sub(r"\b\d{2,4}\b", " tokennumero_medio ", texto)
    texto = re.sub(r"\b\d\b", " tokennumero_pequeno ", texto)
    #texto = re.sub(r"[@#]\w+", " ", texto)
    #ACTUALIZADO: En lugar de eliminar completamente las palabras que comienzan con @ o #, solo eliminamos los símbolos pero mantenemos las palabras.
    #Solo borra los símbolos @ y #, pero deja las palabras intactas
    texto = re.sub(r"[@#]", " ", texto)
    #Elimina caracteres especiales, pero permite conservar los signos de exclamación, el símbolo de dólar y los espacios, ya que pueden ser relevantes para la detección de spam.
    texto = re.sub(r"[^a-záéíóúüñ0-9!$ ]", " ", texto)
    #Elimina espacios adicionales
    doc = nlp(texto)
    #Agregamos 'and not token.is_stop' para filtrar las palabras vacías
    tokens = [token.text for token in doc if not token.is_space and not token.is_stop]
    #Returnamos el texto limpio y tokenizado como una cadena de texto unida por espacios, ya que el vectorizador TF-IDF espera texto en formato de cadena.
    return " ".join(tokens)

#Función para extraer características manuales del mensaje original (antes de limpiar)
#Esto lo usamos para capturar información que podría perderse en la limpieza, como la presencia de URLs, números, signos de exclamación, etc.
def extraer_features(texto):
    #Convertimos el texto a string por si acaso viene en otro formato (aunque debería ser string, esto es una medida de seguridad).
    texto = str(texto)
    #Calculamos la longitud total del mensaje, el número de palabras, el número de dígitos, 
    # la cantidad de URLs (basado en el token que usamos para reemplazar las URLs), 
    # la cantidad de signos de exclamación y la cantidad de símbolos de dólar.
    longitud = len(texto)
    num_palabras = len(texto.split())
    num_numeros = len(re.findall(r'\d', texto))
    num_urls = texto.count("tokenurl")
    num_exclamaciones = texto.count("!")
    num_dolar = texto.count("$")
    #Retornamos estas caracteristicas
    return [longitud, num_palabras, num_numeros, num_urls, num_exclamaciones, num_dolar]

#Lectura del centro de datos
print("Leyendo centro de datos...")
ruta_csv = "../datos/dataset_spam_modelo.csv" 
#Lectura
df = pd.read_csv(ruta_csv)
#Limpiamos el dataset eliminando filas con mensajes vacíos o etiquetas faltantes, y eliminamos mensajes duplicados para evitar sesgos en el entrenamiento del modelo.
df = df.dropna(subset=['mensaje', 'etiqueta'])
df = df.drop_duplicates(subset=['mensaje'])
#Imprime el número total de mensajes en el dataset después de la limpieza, así como el conteo de cada etiqueta (spam vs ham) para entender la distribución de clases.
print(f"Dataset: {len(df)} mensajes | {df['etiqueta'].value_counts().to_dict()}")

#División de datos en entrenamiento y prueba
X_raw, y = df['mensaje'], df['etiqueta']
#Usamos random_state, para que se mantengan valores similares a los documentados
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, stratify=y, random_state=42
)

#Proceso de limpieza y extracción de características

#Se imprime mensaje y se llama la funcion limpiar texto, para los mensajes dividios
print(f"Limpiando {len(X_train_raw)} mensajes de entrenamiento...")
X_train_clean = X_train_raw.apply(limpiar_texto)
print(f"Limpiando {len(X_test_raw)} mensajes de prueba...")
X_test_clean  = X_test_raw.apply(limpiar_texto)
#Después de limpiar, es posible que algunos mensajes hayan quedado vacíos (por ejemplo, si el mensaje original solo contenía una URL o un correo electrónico que fue reemplazado por un token y luego eliminado). 
# Para evitar problemas al vectorizar o entrenar el modelo con mensajes vacíos, 
# filtramos esos casos y mantenemos solo los mensajes que tienen contenido después de la limpieza.
mask_tr = X_train_clean.str.strip() != ""
X_train_clean, y_train, X_train_raw = X_train_clean[mask_tr], y_train[mask_tr], X_train_raw[mask_tr]
mask_te = X_test_clean.str.strip() != ""
X_test_clean, y_test, X_test_raw = X_test_clean[mask_te], y_test[mask_te], X_test_raw[mask_te]

#Obtenemos features manuales
feat_train = np.array([extraer_features(t) for t in X_train_raw])
feat_test  = np.array([extraer_features(t) for t in X_test_raw])

#Se realiza una vectorización de los mensajes limpios usando TF-IDF, 
# limitando a las 5000 palabras más importantes y considerando n-gramas de 1 a 3 palabras para capturar frases comunes que podrían indicar spam.
print("Vectorizando con TF-IDF...")
vec = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9, ngram_range=(1, 3))
X_tr_tfidf = vec.fit_transform(X_train_clean)
X_te_tfidf = vec.transform(X_test_clean)
#Finalmente, combinamos las características numéricas manuales con las características de texto vectorizadas en una sola matriz de características para cada conjunto (entrenamiento y prueba).
X_tr = hstack([X_tr_tfidf, feat_train])
X_te = hstack([X_te_tfidf, feat_test])

#Se define un "menú" de opciones (hiperparámetros) que el modelo va a probar.
param_dist = {
    #Prueba valores de 'C' (penalización) entre 0.01 y 100 usando una escala matemática que favorece probar magnitudes distintas (0.1, 1, 10, etc.)
    "C":            loguniform(1e-2, 1e2),   
    #Prueba dos escenarios: tratar ambas clases igual (None) o darle más peso a los mensajes de spam por ser menos frecuentes ("balanced")
    "class_weight": [None, "balanced"],
    #Prueba si calcular porcentajes de probabilidad (True) afecta el rendimiento frente a no hacerlo (False)
    "probability":  [True, False],
}

#Bloque puramente visual para mostrar en la consola un resumen 
#de lo que el código está a punto de ejecutar.
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


#Crea un modelo SVM "en blanco" usando el kernel RBF, que será la base de las pruebas.
#En este caso se decidio dejar el kerner en rbf, ya que anteriormente se habia encontrado
# que tiene resultados que queremos aproximar, ya que son resultados un poco más reales
#Se puede cambiar al kerner linear.
svm_base = SVC(kernel="rbf")

#Configura la "Validación Cruzada". Cortará los datos en 5 pedazos (n_splits=5), barajándolos (shuffle=True) y asegurando que cada pedazo tenga la misma cantidad de spam/ham (Stratified).
cv = StratifiedKFold(n_splits=5, shuffle=True)

# rea el "Buscador Automático" (RandomizedSearchCV) que hará el trabajo pesado.
busqueda = RandomizedSearchCV(
    estimator=svm_base,         # Usa el modelo SVM que creamos arriba.
    param_distributions=param_dist, # Le pasa el "menú" de opciones que definimos al principio.
    n_iter=40,                  # Le dice que elija 40 combinaciones al azar de ese menú.
    cv=cv,                      # Le dice que use la validación de 5 pedazos para evaluar.
    scoring="f1_macro",         # Le indica que califique cada modelo usando F1 macro (equilibrio entre spam y ham).
    n_jobs=-1,                  # ¡Clave para la velocidad! Le dice a tu computadora que use TODOS los núcleos de su procesador.
    verbose=1,                  # Configura la consola para que te vaya imprimiendo mensajes de cómo va el progreso.
)

#Imprime un aviso al usuario de que el proceso está por empezar.
print("Iniciando búsqueda (esto puede tardar 2-5 minutos)...")

#Toma los datos de entrenamiento (X_tr y y_train) y comienza a probar las 40 combinaciones.
busqueda.fit(X_tr, y_train)

#Muestra de resultados obtenidos
print("\n" + "=" * 60)
print("RESULTADOS DE LA BÚSQUEDA")
print("=" * 60)

#Extrae el diccionario exacto con la combinación ganadora (ej. C=1.5, weight='balanced').
mejor_params = busqueda.best_params_

#Extrae la puntuación más alta (F1 macro) que logró esa combinación ganadora.
mejor_score_cv = busqueda.best_score_

#Imprime en pantalla la configuración ideal que encontró el algoritmo.
print(f"\nMejor combinación encontrada:")
print(f"  C            = {mejor_params['C']:.4f}")  # Imprime 'C' redondeado a 4 decimales.
print(f"  class_weight = {mejor_params['class_weight']}")
print(f"  probability  = {mejor_params['probability']}")

#Imprime la puntuación ganadora multiplicada por 100 para que se lea como porcentaje (ej. 95.50%).
print(f"\nF1 macro en validación cruzada: {mejor_score_cv * 100:.2f}%")

#Imprime un separador para mostrar la tabla de los finalistas.
print("\n--- Top 10 combinaciones ---")

#Convierte el historial completo de las 40 pruebas en una tabla manejable de Pandas.
df_cv = pd.DataFrame(busqueda.cv_results_)

#Filtra la tabla gigante para quedarse solo con las columnas que nos importan: 
# los 3 parámetros probados, el puntaje promedio, la variación (std) y el ranking.
df_cv = df_cv[["param_C", "param_class_weight", "param_probability",
               "mean_test_score", "std_test_score", "rank_test_score"]]

#Ordena la tabla poniendo al #1 hasta arriba y recorta solo las 10 mejores filas.
df_cv = df_cv.sort_values("rank_test_score").head(10)

#Convierte el puntaje promedio (mean) a porcentaje y lo redondea a 2 decimales.
df_cv["mean_test_score"] = (df_cv["mean_test_score"] * 100).round(2)

#Convierte el margen de error / desviación (std) a porcentaje para saber si el modelo es estable.
df_cv["std_test_score"]  = (df_cv["std_test_score"]  * 100).round(2)

#Renombra las columnas con nombres limpios y fáciles de leer para la presentación.
df_cv.columns = ["C", "class_weight", "probability", "F1 CV %", "std %", "rank"]

#Le aplica un formato a toda la columna 'C' para asegurar que todos los números tengan solo 4 decimales.
df_cv["C"] = df_cv["C"].apply(lambda x: f"{x:.4f}")

#Imprime la tabla final en la consola, ocultando los números de fila (index=False) para que quede impecable.
print(df_cv.to_string(index=False))

#Comparativa de modelo base contra modelo optimizado
print("\n" + "=" * 60)
print("COMPARATIVA: BASE vs OPTIMIZADO")
print("=" * 60)

#Modelo base
base = SVC(kernel="rbf", C=1, probability=True, class_weight="balanced")
base.fit(X_tr, y_train)
y_pred_base = base.predict(X_te)

#Modelo optimizado
mejor_modelo = busqueda.best_estimator_
y_pred_mejor = mejor_modelo.predict(X_te)

#Función para calcular e imprimir las métricas de evaluación (accuracy, F1 macro, matriz de confusión) de cada modelo.
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

#Calcula e imprime las métricas de ambos modelos para comparar su rendimiento.
metricas(y_test, y_pred_base,  "Modelo BASE  (C=1, balanced, probability=True)")
metricas(y_test, y_pred_mejor, "Modelo ÓPTIMO (parámetros encontrados por búsqueda)")

print("\n--- Reporte detallado del modelo ÓPTIMO ---")
print(classification_report(y_test, y_pred_mejor))

#Calcula la diferencia en F1 macro entre el modelo optimizado y el modelo base para evaluar si la búsqueda de hiperparámetros realmente mejoró el rendimiento.
f1_base  = f1_score(y_test, y_pred_base,  average='macro') * 100
f1_mejor = f1_score(y_test, y_pred_mejor, average='macro') * 100
diff = f1_mejor - f1_base

#Conclusion 
print("\n" + "=" * 60)
print("CONCLUSIÓN")
print("=" * 60)
#Dependiendo de la diferencia en F1 macro, se imprime una conclusión sobre si la búsqueda de hiperparámetros realmente mejoró el rendimiento del modelo o si el modelo base ya era suficientemente bueno. 
# También se recomienda qué parámetros usar en el script final de entrenamiento basado en los resultados obtenidos.
if diff > 0.5:
    print(f"La búsqueda mejoró el F1 en {diff:.2f} puntos porcentuales.")
    print("Usa estos parámetros en tu script de entrenamiento final.")
elif diff > -0.5:
    print(f"La diferencia es mínima ({diff:+.2f}%). Los parámetros base ya eran buenos.")
    print("Puedes mantener C=1, balanced — la búsqueda lo confirma.")
else:
    print(f"El modelo base supera al optimizado en {-diff:.2f}%.")
    print("Mantén los parámetros originales.")

#Finalmente, se imprime un bloque de código con la configuración ideal que se recomienda usar en el script final de entrenamiento basado en los resultados obtenidos.
print(f"\nParámetros a usar en entrenamiento_corregido.py:")
print(f"  modelo_svm = SVC(")
print(f"      kernel='linear',")
print(f"      C={mejor_params['C']:.4f},")
print(f"      class_weight={repr(mejor_params['class_weight'])},")
print(f"      probability={mejor_params['probability']}")
print(f"  )")