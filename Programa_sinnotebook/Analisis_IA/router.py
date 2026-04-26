#Librerias:
#Fastapi: Framework para construir APIs de manera rápida y sencilla.
from fastapi import APIRouter
#Typing: Para anotaciones de tipos, como List.
from typing import List
#Sys y os: Para manejo de rutas y archivos
import sys
import os
#Importamos el archivo de análisis y el esquema de datos
from Analisis_IA import analyser

from Analisis_IA import schemas

from Modelo_NLP import detector

#Creamos el router para definir las rutas de la API
router = APIRouter()

#Routers para la API

# Endpoint para verificar que el servicio está activo y funcionando correctamente
@router.get("/activo")
async def check_activo():
    return {"status": "activo"}


# Endpoint principal para procesar los mensajes SMS recibidos desde Android.
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

    #1. Convertimos los modelos de Pydantic a diccionarios
    mensajes_crudos = [msg.model_dump() for msg in mensajes]
    
    #2. EL CEREBRO: Filtramos usando el modelo SVM
    spam_detectado = detector.detectar_spam(mensajes_crudos)
    
    #Si el SVM dice que todo está limpio (no hay spam), regresamos una lista vacía
    if not spam_detectado:
        return []

    #3. EL ASISTENTE: Mandamos los mensajes maliciosos a Gemini para generar el resumen final
    spam_resumido = analyser.resumir_spam_detectado(spam_detectado)
    
    return spam_resumido