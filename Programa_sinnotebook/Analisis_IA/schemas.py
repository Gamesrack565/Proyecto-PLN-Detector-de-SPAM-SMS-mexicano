#LIbrerías
#Pydantic es una biblioteca de validación de datos que se utiliza para definir modelos de datos con tipos específicos y validaciones automáticas. 
# En este caso, se están definiendo dos modelos de datos: SmsInput y GrupoSpamOutput.
from pydantic import BaseModel
#Typing es un módulo que proporciona soporte para anotaciones de tipo en Python. 
# En este caso, se está importando List para indicar que el campo mensajesOriginales es una lista de objetos SmsInput.
from typing import List

#1. Lo que recibimos de Android
class SmsInput(BaseModel):
    id: str
    remitente: str
    mensaje: str

#2. Lo que enviaremos de vuelta a Android
class GrupoSpamOutput(BaseModel):
    remitente: str
    resumenIa: str
    mensajesOriginales: List[SmsInput]