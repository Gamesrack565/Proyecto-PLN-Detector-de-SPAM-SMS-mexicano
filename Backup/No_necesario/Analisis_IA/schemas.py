from pydantic import BaseModel
from typing import List

# 1. Lo que recibimos de Android
class SmsInput(BaseModel):
    id: str
    remitente: str
    mensaje: str

# 2. Lo que enviaremos de vuelta a Android
class GrupoSpamOutput(BaseModel):
    remitente: str
    resumenIa: str
    mensajesOriginales: List[SmsInput]