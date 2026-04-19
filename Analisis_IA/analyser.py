import json
import os
import time
from typing import List, Dict, Any
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, InternalServerError
from dotenv import load_dotenv

load_dotenv()

# --- TU GESTOR DE CLAVES INTACTO ---
class KeyManager:
    def __init__(self):
        self.keys = [
            os.getenv("GEMINI_API_KEY_1"),
            os.getenv("GEMINI_API_KEY_2"),
            os.getenv("GEMINI_API_KEY_3")
        ]
        # Filtra las claves que estén vacías
        self.keys = [k for k in self.keys if k]
        
        if not self.keys:
            if os.getenv("GEMINI_API_KEY"):
                self.keys = [os.getenv("GEMINI_API_KEY")]
            else:
                raise ValueError("No hay API Keys configuradas en el archivo .env")
        self.current_index = 0

    def get_current_key(self) -> str:
        return self.keys[self.current_index]

    def switch_key(self) -> str:
        self.current_index = (self.current_index + 1) % len(self.keys)
        return self.get_current_key()

key_manager = KeyManager()

# Modelos disponibles en orden de prioridad (puedes ajustar los nombres exactos)
AVAILABLE_MODELS = ['gemini-2.5-flash-lite', 'gemini-2.5-flash']

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Configuración estricta para evitar errores de sintaxis en el JSON
configuracion_json = genai.GenerationConfig(
    response_mime_type="application/json"
)

def resumir_spam_detectado(mensajes_spam):
    """
    Agrupa los mensajes SPAM y pide a Gemini un resumen estructurado.
    Implementa rotación automática de API Keys y Modelos en caso de agotar la cuota.
    """
    
    if not mensajes_spam:
        return []

    texto_mensajes = json.dumps(mensajes_spam, ensure_ascii=False, indent=2)

    prompt = f"""
    Eres un experto en ciberseguridad. Los siguientes mensajes ya fueron clasificados como SPAM por un modelo de Machine Learning.
    
    Tu tarea es agrupar los mensajes que provengan del mismo 'remitente' o que traten sobre la misma estafa, y generar un 'resumenIa' corto advirtiendo al usuario de qué trata y agrega una explicacion corta sobre fraudes similares.

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

    # Bucle 1: Iterar sobre los modelos disponibles
    for nombre_modelo in AVAILABLE_MODELS:
        print(f"\n🔄 Intentando usar el modelo: {nombre_modelo}")
        
        # Bucle 2: Iterar sobre las API Keys disponibles
        for intento in range(len(key_manager.keys)):
            current_key = key_manager.get_current_key()
            
            try:
                # 1. Configurar la librería con la clave actual
                genai.configure(api_key=current_key)
                
                # 2. Inicializar el modelo explícitamente en cada intento
                modelo_actual = genai.GenerativeModel(
                    model_name=nombre_modelo,
                    generation_config=configuracion_json,
                    safety_settings=SAFETY_SETTINGS
                )
                
                # 3. Llamar a la API
                respuesta = modelo_actual.generate_content(prompt)
                texto_respuesta = respuesta.text

                # 4. Limpiar y procesar la respuesta
                texto_respuesta = texto_respuesta.strip()
                if texto_respuesta.startswith("```json"):
                    texto_respuesta = texto_respuesta[7:]
                if texto_respuesta.startswith("```"):
                    texto_respuesta = texto_respuesta[3:]
                if texto_respuesta.endswith("```"):
                    texto_respuesta = texto_respuesta[:-3]
                    
                resultado_json = json.loads(texto_respuesta.strip())
                return resultado_json

            # Si nos quedamos sin peticiones (Error de Cuota 429)
            except ResourceExhausted:
                print(f"⚠️ Cuota agotada para la API Key {key_manager.current_index + 1}. Cambiando clave...")
                key_manager.switch_key()
                time.sleep(1) # Pausa breve para no saturar los servidores de Google
                continue # Pasa a la siguiente clave (Bucle 2)
            
            # Si hay un error interno de Google o de conexión
            except InternalServerError:
                print("⚠️ Error en los servidores de Google. Reintentando...")
                time.sleep(2)
                continue
                
            # Si hay un error de conversión JSON u otra falla crítica
            except json.JSONDecodeError as e:
                print(f"❌ Error al decodificar JSON de Gemini: {e}")
                return []
            except Exception as e:
                print(f"❌ Error inesperado: {str(e)}")
                break # Rompe el ciclo de las claves y pasa al siguiente modelo
                
        print(f"❌ Se agotaron todas las claves para el modelo {nombre_modelo}.")
        # Si termina el bucle de claves, el código continúa y el Bucle 1 pasa al siguiente modelo

    # Si se agotan TODOS los modelos y TODAS las claves
    print("🚨 FATAL ERROR: Se agotaron las cuotas de todos los modelos y claves.")
    return []