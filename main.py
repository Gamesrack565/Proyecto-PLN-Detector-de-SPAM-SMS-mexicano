import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Analisis_IA import router as ai_router

# Configuración de la aplicación FastAPI
app = FastAPI(
    title="Detector de Spam - Backend",
    version="2.5.0",
    description="API para procesar SMS enviados desde Android y detectar Spam y resumen de IA."
)

# Configuración de CORS
origins = ["*"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    ai_router.router,
    prefix="/api/sms",
    tags=["Análisis Spam SMS"]
)

@app.get("/")
def read_root():
    return {"message": "API de Detección de Spam funcionando correctamente."}