#Librerias:

#Json: Es una biblioteca estándar de Python que se utiliza para trabajar con datos en formato JSON.
import json
#os: Proporciona funciones de control de archivos
import os
#time: Agrega una pausa entre solicitudes para no saturar la API de Google
import time
#typing: Proporciona soporte para anotaciones de tipo en Python, lo que ayuda a mejorar la legibilidad y el mantenimiento del código.
from typing import List, Dict, Any
#Google Generative AI: Es la biblioteca oficial de Google para interactuar con sus modelos de generación de texto, como Gemini. Permite enviar solicitudes a la API de Google y recibir respuestas generadas por el modelo.
import google.generativeai as genai
#Google API Core Exceptions: Proporciona clases de excepciones para manejar errores específicos que pueden ocurrir al interactuar con las APIs de Google, como ResourceExhausted (cuando se exceden los límites de uso) e InternalServerError (cuando ocurre un error interno en el servidor).
from google.api_core.exceptions import ResourceExhausted, InternalServerError
#Dotenv: Es una biblioteca que carga variables de entorno desde un archivo .env, lo que permite gestionar de manera segura las credenciales y configuraciones sensibles sin hardcodearlas en el código fuente.
from dotenv import load_dotenv

load_dotenv()
#Clase para manejar las claves de API de Google Gemini, permitiendo rotar entre varias claves para evitar problemas de límite de uso y garantizar un acceso continuo a la API.
class Control_Llaves:
    def __init__(self):
        #Definicion de las 3 claves API que se utilizarán.
        #Esto se hace, con la finalidad de rotar entre proyectos gratuitos, para en caso de que se alcance el límite de uso en una clave, poder cambiar a otra sin interrumpir el servicio.
        self.llaves = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3")
        ]
        #Filtra las claves que estén vacías o sean None
        self.llaves = [k for k in self.llaves if k]
        #Indica que se iniciara con la primera clave de la lista
        self.inicio_ind = 0
    #Obtiene la clave actual que se está utilizando para las solicitudes a la API de Google Gemini.
    def obtener_llave(self) -> str:
        return self.llaves[self.inicio_ind]
    #Cambio de clave: Esta función rota a la siguiente clave en la lista cada vez que se llama, asegurando que se utilicen todas las claves disponibles de manera equitativa. Si se alcanza el final de la lista, vuelve al inicio.
    def cambio_llave(self) -> str:
        self.inicio_ind = (self.inicio_ind + 1) % len(self.llaves)
        return self.obtener_llave()

Control_Llaves = Control_Llaves()

#Modelos que se utilizarán para generar las respuestas a las solicitudes de clasificación de mensajes SMS.
MODELOS_UTILIZAR = ['gemini-2.5-flash-lite', 'gemini-2.5-flash']

PARAMETROS_SEGURIDAD = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

#Configuración estricta para evitar errores de sintaxis en el JSON
configuracion_json = genai.GenerationConfig(
    response_mime_type="application/json"
)

#Funcion principal que se encarga de resumir los mensajes SPAM detectados utilizando Google Gemini, con manejo avanzado de errores y rotación de claves/modelos.
def resumir_spam_detectado(mensajes_spam):
    """
    Agrupa los mensajes SPAM y pide a Gemini un resumen estructurado.
    Implementa rotación automática de API Keys y Modelos en caso de agotar la cuota o fallar.
    """
    #Si no hay mensajes SPAM, retorna una lista vacía
    if not mensajes_spam:
        return []
    #Convierte la lista de mensajes SPAM a un formato JSON legible para incluirlo en el prompt de Gemini
    texto_mensajes = json.dumps(mensajes_spam, ensure_ascii=False, indent=2)
    #Prompt a utilizar:
    prompt = f"""
    Eres un experto en detector de spam. Los siguientes mensajes ya fueron clasificados como SPAM por un modelo de Machine Learning.
    
    Tu tarea es agrupar los mensajes que provengan del mismo 'remitente', y generar un 'resumenIa' indicandole al usuario porque fueron marcados como SPAM y en caso del que mensaje tenga indicios de una estafa indicarselo. DEBER SER UN RESUMEN SUMAMENTE BREVE
    En caso de que notes mensajes con codigos de acceso, solo le indicaras de que son y que verifique si realmente los ha solicitado

    IMPORTANTE: Tu respuesta debe ser ÚNICAMENTE un arreglo JSON válido con esta estructura exacta:
    [
      {{
        "remitente": "Nombre del estafador o banco falso",
        "resumenIa": "Advertencia breve sobre la estafa",
        "mensajesOriginales": [
          {{
            "id": "1",
            "remitente": "...",
            "mensaje": "..."
          }}
        ]
      }}
    ]

    Mensajes a procesar:
    {texto_mensajes}
    """

    #Bucle 1: Iterar sobre los modelos disponibles
    for nombre_modelo in MODELOS_UTILIZAR:
        print(f"\n🔄 Intentando usar el modelo: {nombre_modelo}")
        
        #Bucle 2: Iterar sobre las API Keys disponibles
        for intento in range(len(Control_Llaves.llaves)):
            current_key = Control_Llaves.obtener_llave()
            
            try:
                #1. Configurar la librería con la clave actual
                genai.configure(api_key=current_key)
                
                #2. Inicializar el modelo
                modelo_actual = genai.GenerativeModel(
                    model_name=nombre_modelo,
                    generation_config=configuracion_json,
                    safety_settings=PARAMETROS_SEGURIDAD
                )
                
                #3. Llamar a la API
                respuesta = modelo_actual.generate_content(prompt)
                
                #Manejo de respuesta bloqueada por seguridad.
                if not respuesta.parts:
                    print("ADVERTENCIA: Respuesta bloqueada por filtros de seguridad.")
                    raise ValueError("Respuesta vacía o bloqueada.")
                #Extraer el texto de la respuesta
                texto_respuesta = respuesta.text.strip()

                #4. Limpiar formato Markdown si viene incluido
                if texto_respuesta.startswith("```json"):
                    texto_respuesta = texto_respuesta[7:]
                if texto_respuesta.startswith("```"):
                    texto_respuesta = texto_respuesta[3:]
                if texto_respuesta.endswith("```"):
                    texto_respuesta = texto_respuesta[:-3]
                    
                #Si llega hasta aquí sin errores, retorna el JSON y corta los bucles (¡Éxito!)
                resultado_json = json.loads(texto_respuesta.strip())
                return resultado_json

            #Si nos quedamos sin cuota (Error 429) -> Cambiamos de Key, no de modelo
            except ResourceExhausted:
                print(f"ADVERTENCIA: Cuota agotada para la API Key {Control_Llaves.inicio_ind + 1}. Cambiando clave...")
                Control_Llaves.switch_key()
                time.sleep(1) 
                continue 
            
            #Si hay un error interno del servidor -> Reintentamos con la misma Key/Modelo
            except InternalServerError:
                print("ADVERTENCIA: Error en los servidores de Google. Reintentando...")
                time.sleep(2)
                continue
                
            #Si hay error de JSON -> El modelo "lite" falló, saltamos al modelo más capaz
            except json.JSONDecodeError as e:
                print(f"ERROR: El modelo {nombre_modelo} generó un JSON inválido: {e}. Probando siguiente modelo...")
                break # <-- Rompe el ciclo de claves, salta al Bucle 1 (siguiente modelo)
            
            #Si el modelo no existe o tira cualquier otro error grave -> Saltamos al siguiente modelo
            except Exception as e:
                print(f"ERROR CRÍTICO con {nombre_modelo}: {str(e)}. Descartando modelo...")
                break # <-- Rompe el ciclo de claves, salta al Bucle 1 (siguiente modelo)
                
        # Si sale del Bucle 2 de forma natural (se agotaron las claves para este modelo), imprime y sigue
        print(f"--> Se agotaron las opciones o falló definitivamente el modelo {nombre_modelo}.")

    # Si se agotan TODOS los modelos y TODAS las claves, devolvemos lista vacía
    print("FATAL ERROR: Se agotaron las cuotas de todos los modelos o hubo fallos en cadena.")
    return []