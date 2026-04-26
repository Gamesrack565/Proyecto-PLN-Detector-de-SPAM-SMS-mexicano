##Librerias a utilizar:

#Pandas: Para la manipulación de datos.
import pandas as pd

#Requests: Para realizar solicitudes HTTP y descargar archivos.
import requests

#io: Para manejar flujos de datos en memoria.
import io

#re: Para trabajar con expresiones regulares.
import re

#os: Para interactuar con el sistema operativo, como crear directorios.
import os

#time: Para pausar el tiempo entre solicitudes y evitar sobrecargar la api de google translate.
import time

#tqdm: Para mostrar barras de progreso durante la traduccion del texto.
from tqdm import tqdm

#deep_translator: Para traducir texto utilizando el servicio de Google Translate.
from deep_translator import GoogleTranslator

#Se tiene el centro de datos Mexicano, descargado de manera local, el cual se encuentra en formato CSV.
#Obtenido de la siguiente página: https://www.kaggle.com/datasets/aldoivan/sms-spam-mexico-dataset-en-espaol-mexicano
RUTA_DATASET_MEXICO = os.path.join("../..", "dataset.csv")

#PASO 1
#CARGA DEL CENTRO DE DATOS MEXICANO

#Impresion de mensaje para indicar que se esta cargando el centro de datos.
print("=" * 60)
print("PASO 1: Cargando dataset México (Kaggle)...")
print("=" * 60)

#Carga del centro de datos mexicano desde el archivo CSV.
df_mexico = pd.read_csv(RUTA_DATASET_MEXICO)

#Normalizar columnas a: mensaje, etiqueta
#Esto como medida de seguridad, por si se ingresa un centro de datos con columnas diferentes a las esperadas.
#Se puede quitar, ya que conocemos el formato del dataset.
df_mexico = df_mexico.rename(columns={
    "texto": "mensaje",
    "text":  "mensaje",
    "sms":   "mensaje",
})
#Se seleccionan solo las columnas de mensaje y etiqueta, y se normalizan las etiquetas a minúsculas y sin espacios.
df_mexico = df_mexico[["mensaje", "etiqueta"]].copy()
#Se normalizan las etiquetas a minúsculas y sin espacios, para evitar problemas de clasificación debido a diferencias en el formato de las etiquetas.
df_mexico["etiqueta"] = df_mexico["etiqueta"].str.strip().str.lower()
#Se agrega una columna de fuente para identificar el origen de los datos, lo cual es útil para el análisis posterior y para evitar confusiones al combinar con otros centros de datos.
df_mexico["fuente"] = "kaggle_mexico"

#Se imprime un mensaje indicando que se ha cargado el centro de datos mexicano, junto con la cantidad de mensajes y la distribución de etiquetas.
print(f"  Cargados: {len(df_mexico)} mensajes")
print(f"  {df_mexico['etiqueta'].value_counts().to_dict()}")

#PASO 2
#DESCARGA DEL CENTRO DE DATOS ESPAÑOL DESDE HUGGING FACE
#Este es un centro de datos extra en español, que se ha encontrado en Hugging Face, 
# y que se ha decidido incluir para aumentar la cantidad de datos disponibles para el entrenamiento del modelo.

#Se imprime un mensaje para indicar que se esta descargando el centro de datos español desde Hugging Face.
print("\n" + "=" * 60)
print("PASO 2: Descargando el centro de datos español (Hugging Face)...")
print("=" * 60)

#En Hugging Face, el centro de datos se encuentra dividido en dos partes: train.csv y test.csv, por lo que se definen las URLs de ambas partes para descargarlas posteriormente.
HF_URLS = [
    "https://huggingface.co/datasets/softecapps/spam_ham_spanish/resolve/main/train.csv",
    "https://huggingface.co/datasets/softecapps/spam_ham_spanish/resolve/main/test.csv",
]

#Se crea una lista vacía para almacenar las partes del centro de datos descargadas desde Hugging Face, y se itera sobre las URLs para descargarlas y cargarlas en dataframes de pandas.
partes_hf = []
#Ciclo para descargar cada parte del centro de dattos
for url in HF_URLS:
    try:
        #Se realiza una petición HTTP para descargar el archivo
        #Cuenta con un tiempo de espera de 30 segundos.
        resp = requests.get(url, timeout=30)
        #Si la respuesta no es exitosa, se lanza una excepción para manejar el error.
        resp.raise_for_status()
        #Convierte el texto descargado en un dataframe de pandas utilizando io.StringIO para manejar el flujo de datos en memoria.
        df_part = pd.read_csv(io.StringIO(resp.text))
        #Elimina espacios en blanco de los nombres de las columnas para evitar problemas al acceder a ellas posteriormente.
        df_part.columns = df_part.columns.str.strip()
        #Agrega la parte descargada a la lista de partes del centro de datos de Hugging Face.
        partes_hf.append(df_part)
        #Imprime un mensaje indicando que se ha descargado la parte del centro de datos, junto con la cantidad de filas que contiene.
        print(f"  ✓ Descargado: {url.split('/')[-1]} — {len(df_part)} filas")
    #En caso de que no se pueda descargar, marca un error
    except Exception as e:
        print(f"  ✗ No se pudo descargar {url.split('/')[-1]}: {e}")

#Si al menos una descarga fue exitosa, se procede a combinar las partes descargadas en un solo marco de datos, 
# normalizar las columnas para asegurarse de que tengan los nombres esperados (mensaje y etiqueta), 
# y normalizar las etiquetas al formato spam/ham para mantener la consistencia con el centro de datos mexicano.
if partes_hf:
    #Une las partes descargadas.
    df_hf = pd.concat(partes_hf, ignore_index=True)
    #Normalizar columnas (puede llamarse 'mensaje'/'tipo' o 'Mensaje'/'Etiqueta')
    df_hf.columns = df_hf.columns.str.strip().str.lower()
    #Busca las columnas de texto y etiqueta, permitiendo diferentes nombres comunes para cada una.
    col_texto = next((c for c in df_hf.columns if c in ["mensaje", "text", "sms", "message"]), None)
    #Busca las columnas de etiqueta, permitiendo diferentes nombres comunes para cada una.
    col_label = next((c for c in df_hf.columns if c in ["tipo", "etiqueta", "label"]), None)

    #Si se encuentran las columnas de texto y etiqueta, se seleccionan y renombrar a mensaje y 
    # etiqueta respectivamente, se normalizan las etiquetas al formato spam/ham, 
    # se agrega una columna de fuente para identificar el origen de los datos, 
    # y se imprime un mensaje indicando la cantidad de mensajes descargados y la distribución de etiquetas.
    if col_texto and col_label:
        #Selecciona solo las columnas de texto y etiqueta, y las renombra a mensaje y etiqueta respectivamente.
        df_hf = df_hf[[col_texto, col_label]].rename(
            columns={col_texto: "mensaje", col_label: "etiqueta"}
        )
        #Normaliza las etiquetas a minúsculas y sin espacios, para evitar problemas de clasificación debido a diferencias en el formato de las etiquetas.
        df_hf["etiqueta"] = df_hf["etiqueta"].str.strip().str.lower()
        #Normalizar etiquetas al formato spam/ham
        df_hf["etiqueta"] = df_hf["etiqueta"].replace({"legítimo": "ham", "legitimo": "ham"})
        #Agrega una columna de fuente para identificar el origen de los datos, lo cual es útil para el análisis posterior y para evitar confusiones al combinar con otros centros de datos.
        df_hf["fuente"] = "huggingface_es"
        #Imprime un mensaje indicando la cantidad de mensajes descargados y la distribución de etiquetas.
        print(f"  Total HF: {len(df_hf)} mensajes — {df_hf['etiqueta'].value_counts().to_dict()}")
    else:
        #En caso de algun error se indica.
        print(f"  ✗ Columnas inesperadas: {df_hf.columns.tolist()} — se omite esta fuente")
        df_hf = pd.DataFrame(columns=["mensaje", "etiqueta", "fuente"])
else:
    #En caso de algun error, se indica.
    print("  ✗ No se pudo descargar el centro de datos de HuggingFace — se continúa sin él")
    df_hf = pd.DataFrame(columns=["mensaje", "etiqueta", "fuente"])

#PASO 3
#DESCARGA Y TRADUCCIÓN DEL CENTRO DE DATOS UCI (INGLÉS)
#Este es un centro de datos en inglés, que se ha decidido incluir para aumentar la cantidad de datos disponibles 
# para el entrenamiento del modelo, aunque se traducirá al español para mantener la consistencia con los otros centros de datos.

#Se imprime un mensaje para indicar que se esta descargando
print("\n" + "=" * 60)
print("PASO 3: Descargando y traduciendo centro de datos UCI (inglés)...")
print("=" * 60)

#URL del centro de datos UCI, que se encuentra en inglés y se descargará para luego traducirlo al español.
UCI_URL = "https://raw.githubusercontent.com/mohitgupta-1O1/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"

#Se intenta descargar el centro de datos
try:
    #Se realiza una peticion HTTP para descargar el archivo, con un tiempo de espera de 30 segundos.
    resp = requests.get(UCI_URL, timeout=30)
    #Si la respuesta no es exitosa, se lanza una excepción para manejar el error.
    resp.raise_for_status()
    #Convierte el texto descargado en un marco de datos de pandas utilizando io.StringIO para manejar el flujo de datos en memoria,
    # y se especifica la codificación latin-1 para evitar problemas con caracteres especiales.
    df_uci_raw = pd.read_csv(
        io.StringIO(resp.text),
        encoding="latin-1",
        usecols=[0, 1],
        names=["etiqueta", "mensaje"],
        header=0,
    )
    #Se normalizan las etiquetas a minúsculas y sin espacios, para evitar problemas de clasificación debido a diferencias en el formato de las etiquetas.
    df_uci_raw["etiqueta"] = df_uci_raw["etiqueta"].str.strip().str.lower()
    #Se imprime un mensaje indicando que se ha descargado el centro de datos UCI, junto con la cantidad de mensajes y la distribución de etiquetas.
    print(f"  UCI descargado: {len(df_uci_raw)} mensajes")
    print(f"  {df_uci_raw['etiqueta'].value_counts().to_dict()}")

    #Tomamos SOLO mensajes de smishing/phishing más ambiguos:
    #spam que contenga URLs, números de teléfono o palabras de alerta
    #COMO EXTRA: Para que los resultados se mantengan fijos se puso una semilla para la seleccion aleatoria
    patron_ambiguo = r'(http|www|call|claim|free|win|prize|urgent|verify|account|bank)'
    df_spam_uci = df_uci_raw[
        (df_uci_raw["etiqueta"] == "spam") &
        (df_uci_raw["mensaje"].str.contains(patron_ambiguo, case=False, regex=True))
    ].sample(n=min(200, len(df_uci_raw[df_uci_raw["etiqueta"]=="spam"])), random_state=42)

    # Esto es opcional, pero se decidió tomar solo mensajes ham más cortos y conversacionales, ya que son más difíciles de clasificar y aportan más valor al entrenamiento del modelo.
    #COMO EXTRA: Para que los resultados se mantengan fijos se puso una semilla para la seleccion aleatoria
    df_ham_uci = df_uci_raw[
        (df_uci_raw["etiqueta"] == "ham") &
        (df_uci_raw["mensaje"].str.len() < 120)
    ].sample(n=min(200, len(df_uci_raw[df_uci_raw["etiqueta"]=="ham"])), random_state=42)

    #Se combinan los mensajes seleccionados de spam y ham en un solo marco de datos para su posterior traducción.
    df_uci_muestra = pd.concat([df_spam_uci, df_ham_uci], ignore_index=True)
    #Se imprime un mensaje indicando la cantidad de mensajes seleccionados para traducir.
    print(f"  Muestra seleccionada para traducir: {len(df_uci_muestra)} mensajes")

    #Traducción con GoogleTranslator (sin API key, límite 5000 chars por llamada)
    print("  Traduciendo al español (esto tarda ~3-5 minutos, ten paciencia)...")
    #Traduccion de texto de inglés a español
    translator = GoogleTranslator(source="en", target="es")

    #Funcion para traducir un texto, con reintento en caso de error.
    def traducir_seguro(texto):
        """Traduce un texto, con reintento en caso de error."""
        try:
            #Google Translate tiene límite de 5000 chars
            #Si el texto es muy largo, se recorta para evitar errores de traducción, aunque en este caso no debería ser un problema ya que los mensajes de texto suelen ser cortos.
            if len(texto) > 4500:
                #Recorta el texto a 4500 caracteres para evitar errores de traducción.
                texto = texto[:4500]
            #Realiza la traducción utilizando Google Translator.
            resultado = translator.translate(texto)
            #pequeña pausa para no saturar la API gratuita
            time.sleep(0.15)
            #Retorna el resultado de la traducción, o el texto original si la traducción falla por alguna razón 
            # (aunque en este caso se espera que siempre funcione, a menos que haya un error de conexión o algo similar).
            return resultado if resultado else texto
        #En caso de que falle la traducción, se espera un segundo y se intenta traducir nuevamente, para manejar posibles errores temporales de conexión o de la API.
        except Exception:
            time.sleep(1)
            try:
                return translator.translate(texto[:2000])
            except Exception:
                #Si falla dos veces, dejamos el original
                return texto   

    #Se crea una lista vacía para almacenar las traducciones de los mensajes, y se itera sobre cada mensaje en 
    # la muestra seleccionada para traducirlo utilizando la función traducir_seguro, 
    # mostrando una barra de progreso con tqdm para indicar el avance de la traducción.
    traducciones = []
    #Ciclo para traducir y muestra de progreso con tqdm para indicar el avance de la traducción.
    for msg in tqdm(df_uci_muestra["mensaje"], desc="  Traduciendo", unit="msg"):
        traducciones.append(traducir_seguro(msg))

    #Se crea un nuevo marco de datos con los mensajes traducidos, manteniendo las etiquetas originales 
    # y agregando una columna de fuente para identificar el origen de los datos, lo cual es útil para el análisis posterior 
    # y para evitar confusiones al combinar con otros centros de datos.
    df_uci_traducido = df_uci_muestra.copy()
    df_uci_traducido["mensaje"] = traducciones
    df_uci_traducido["fuente"] = "uci_traducido"
    #Imprime un mensaje indicando que se ha completado la traducción, junto con la cantidad de mensajes traducidos.
    print(f"  ✓ Traducción completada: {len(df_uci_traducido)} mensajes")

#En caso de que no se pueda descargar o traducir el centro de datos UCI, se maneja la excepción y se crea un marco de datos vacío para evitar problemas en los pasos posteriores.
except Exception as e:
    print(f"  ✗ Error descargando UCI: {e}")
    df_uci_traducido = pd.DataFrame(columns=["mensaje", "etiqueta", "fuente"])

#PASO 4
#FUSIONAR Y LIMPIAR

print("\n" + "=" * 60)
print("PASO 4: Fusionando y limpiando...")
print("=" * 60)

#Se realiza la union de los tres centros de datos (mexicano, Hugging Face en español, y UCI traducido),
df_final = pd.concat(
    [df_mexico, df_hf, df_uci_traducido],
    ignore_index=True
    #Selecciona solo las columnas de mensaje, etiqueta y fuente, para mantener solo la información relevante para el entrenamiento del modelo.
)[["mensaje", "etiqueta", "fuente"]]

#Se imprime la cantidad total de mensajes antes de realizar una limpieza.
print(f"  Total antes de limpiar: {len(df_final)}")

#Elimina nulos y vacíos
df_final = df_final.dropna(subset=["mensaje", "etiqueta"])
df_final = df_final[df_final["mensaje"].str.strip() != ""]

#Normaliza etiquetas (por si hay variantes)
df_final["etiqueta"] = df_final["etiqueta"].str.strip().str.lower()
etiquetas_validas = {"spam", "ham"}
#Destruye cualquier fila que no sean las etiquetas válidas.
df_final = df_final[df_final["etiqueta"].isin(etiquetas_validas)]

#Elimina duplicados exactos
#Calcula la cantidad de mensajes antes de eliminar duplicados para poder mostrar cuántos se eliminaron.
antes = len(df_final)
#Se eliminan los mensajes duplicados exactos, es decir, aquellos que tienen el mismo texto en la columna de mensaje, 
# independientemente de su etiqueta o fuente. 
# Esto ayuda a reducir el ruido en el centro de datos y a evitar que el modelo aprenda patrones basados en mensajes repetidos.
df_final = df_final.drop_duplicates(subset=["mensaje"])
#Muestra cuantos mensajes eran duplicados exactos y fueron eliminados.
print(f"  Duplicados exactos eliminados: {antes - len(df_final)}")

#Se imprime un resumen final del centro de datos después de la limpieza, 
# incluyendo la cantidad total de mensajes, la distribución por etiqueta (spam/ham) 
# y la distribución por fuente (origen del mensaje).
print(f"\n  Total final: {len(df_final)} mensajes")
print(f"  Por etiqueta:\n{df_final['etiqueta'].value_counts().to_string()}")
print(f"\n  Por fuente:\n{df_final['fuente'].value_counts().to_string()}")

#PASO 5
#GUARDAR

print("\n" + "=" * 60)
print("PASO 5: Guardando...")
print("=" * 60)

#Se guarda el centro de datos final en un archivo CSV, sin incluir el índice y utilizando la codificación utf-8-sig 
# para asegurar la compatibilidad con caracteres especiales.
SALIDA = "dataset_spam_final.csv"
df_final.to_csv(SALIDA, index=False, encoding="utf-8-sig")
#Imprime un mensaje indicando que se ha guardado correctamente
print(f"  ✓ Guardado en: {SALIDA}")

#También guarda una versión solo con las columnas que espera el modelo
df_modelo = df_final[["mensaje", "etiqueta"]].copy()
df_modelo.columns = ["mensaje", "etiqueta"]
df_modelo.to_csv("dataset_spam_modelo.csv", index=False, encoding="utf-8-sig")
#Imprime un mensaje indicando que se ha guardado correctamente
print(f"  ✓ Versión para el modelo: dataset_spam_modelo.csv")