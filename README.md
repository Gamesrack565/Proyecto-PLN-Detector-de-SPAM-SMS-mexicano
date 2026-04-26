# Proyecto PLN: Detector de SPAM SMS (Contexto Mexicano)

Una solución integral de ciberseguridad móvil para la detección, filtrado y análisis de mensajes SMS maliciosos (Spam/Phishing/Smishing) adaptada a los modismos, marcas y formatos de México. 

Este proyecto combina Procesamiento de Lenguaje Natural (PLN), Machine Learning y modelos generativos para ofrecer una capa de protección inteligente directamente en el dispositivo del usuario.

## Características Principales

* **Análisis Activo (Android):** Aplicación móvil capaz de leer la bandeja de entrada y evaluar los mensajes para mantenerla limpia.
* **Clasificación Inteligente:** Backend impulsado por un modelo de Machine Learning (Máquinas de Soporte Vectorial) entrenado específicamente con un dataset balanceado de SMS en español mexicano.
* **Agrupación y Resumen con IA:** Integración con modelos de lenguaje de gran tamaño (LLMs) que agrupa las amenazas por estafador/remitente y genera un reporte breve explicando el riesgo (ej. *Intento de robo de cuenta*, *Phishing bancario*).

## Arquitectura y Tecnologías

El sistema está dividido en un ecosistema Cliente-Servidor:

**1. Frontend Móvil (Android):**
* Aplicación nativa con interfaz declarativa y moderna.
* Gestión de tareas en segundo plano (WorkManager) y bases de datos locales (Room) para persistencia del spam detectado.

**2. Backend & Motor NLP (Python):**
* **API REST:** Construida con FastAPI para procesar peticiones móviles con baja latencia.
* **Pipeline PLN:** Limpieza de texto con expresiones regulares y vectorización mediante TF-IDF.
* **Clasificador:** Modelo SVM (`scikit-learn`) con kernel lineal.
* **Agente IA:** Conexión con la API de Google Gemini, implementando un sistema robusto de rotación de *API Keys* y manejo de cuotas para asegurar disponibilidad continua.

## Cómo Funciona el Flujo

1. El dispositivo Android recibe un nuevo SMS o el usuario solicita un escaneo manual.
2. La aplicación extrae el texto y lo envía empaquetado vía HTTP a la API REST.
3. El pipeline de PLN vectoriza el mensaje y el modelo predice si es un aviso legítimo (*Ham*) o una amenaza (*Spam*).
4. Si se detectan amenazas, los mensajes se envían al agente de IA Generativa, el cual extrae el contexto y devuelve un resumen JSON estructurado.
5. La app de Android recibe la alerta, bloquea visualmente el mensaje y notifica al usuario con la explicación del riesgo.

## Construcción del centro de datos
Para entrenar este modelo se construyó un corpus especializado fusionando múltiples fuentes:
* Datasets públicos de Kaggle etiquetados en español de México.
* Mensajes extraídos de Hugging Face.
* Traducción automática e inyección manual de plantillas de estafas corporativas recientes (Paqueterías falsas, códigos 2FA, promociones telefónicas engañosas).
