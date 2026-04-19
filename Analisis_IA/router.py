from fastapi import APIRouter
from typing import List
import sys
import os

from . import schemas, analyser

# Para poder importar la carpeta Modelo_NLP que está al mismo nivel
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Modelo_NLP import detector

router = APIRouter()

@router.post("/clasificar-sms", response_model=List[schemas.GrupoSpamOutput])
async def procesar_mensajes_sms(mensajes: List[schemas.SmsInput]):
    """
    Flujo Real: 
    1. Recibe todos los SMS.
    2. Pasa por el modelo SVM local para filtrar solo el SPAM.
    3. Manda el SPAM a Gemini para agruparlo y resumirlo.
    """
    if not mensajes:
        return []

    # 1. Convertimos los modelos de Pydantic a diccionarios
    mensajes_crudos = [msg.model_dump() for msg in mensajes]
    
    # 2. EL CEREBRO: Filtramos usando tu modelo de Machine Learning (SVM)
    spam_detectado = detector.detectar_spam(mensajes_crudos)
    
    # Si el SVM dice que todo está limpio (no hay spam), regresamos una lista vacía
    if not spam_detectado:
        return []

    # 3. EL ASISTENTE: Mandamos los mensajes maliciosos a Gemini para generar el resumen final
    spam_resumido = analyser.resumir_spam_detectado(spam_detectado)
    
    return spam_resumido