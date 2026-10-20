# Sistema de Predicción de Deserción Estudiantil \- Grupo 01

Trabajo final de Tecnologías Emergentes

Este proyecto consiste en un sistema de predicción de éxito académico y riesgo de deserción estudiantil. Utiliza modelos de Machine Learning consumidos a través de una API REST y un Chatbot interactivo de reglas.

## Instalación y Ejecución (Reproducción desde cero)

1. Usar **Python 3.13** y **scikit-learn 1.9.1**, crear un entorno virtual (`python -m venv .venv`) y activarlo; instalar las dependencias:  
     
   python \-m pip install \-r requirements.txt  
     
2. El archivo **`data/data.csv` ya viene incluido** en esta entrega. Si no está presente, obtener el conjunto **Predict Students’ Dropout and Academic Success** desde [UCI, dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), colocar el CSV original en `data/data.csv` y comprobar su formato: 4424 registros y separador `;`.  
3. Conservar `modelos/*.joblib`, `modelos/fichas_json/*.json` y `modelos/mejor_version.json` para **reproducir la inferencia del modelo ya promovido** sin reentrenar.

### Ejecución de Pruebas

python \-m pytest \-q

*(Nota: Actualmente pasan 76/76 pruebas aprobadas en Python 3.13 con scikit-learn 1.9.1 y las dependencias fijadas).*

### Levantar la API


Desde la raíz del proyecto:

uvicorn src.api:app \--reload

La API expone `GET /salud`, `POST /predecir` y documentación interactiva en `/docs`. Usa automáticamente la versión indicada en `modelos/mejor_version.json`.

**Ejemplo de JSON para `POST /predecir`:**

{

 "unidades\_inscritas\_s1": 6,

 "unidades\_aprobadas\_s1": 5,

 "promedio\_s1": 12.5,

 "matricula\_al\_dia": 1,

 "deudor": 0,

 "becario": 1,

 "edad\_ingreso": 19,

 "nota\_admision": 130.0

}

### Ejecutar el Chatbot (Requiere la API levantada)

En una terminal separada, ejecutar:

python \-m src.chatbot

El chatbot interactúa secuencialmente solicitando los datos del estudiante y sugiere acudir al tutor si identifica `Dropout` o fallas del servicio.

## Estado Actual y Limitaciones Conocidas

* **Entorno y Dependencias:** La suite completa funciona exitosamente en Python 3.13 y scikit-learn 1.9.1, garantizando compatibilidad total con los archivos `.joblib` guardados.  
* **Evaluación del Modelo:** La cifra **0.6514 corresponde a macro-F1 de validación**. La ficha del modelo promovido (`regresion_logistica`) registra **0.6537 de macro-F1 de prueba**.  
* **Regla de Decisión:** Se conservó la regla de decisión sin umbral probabilístico. No se debe fijar un umbral sin validación previa.

**⚠️ IMPORTANTE SOBRE EL REENTRENAMIENTO:** No volver a ejecutar `entrenar.py` ni `entrenar.py --final` sobre la selección entregada. La prueba final ya fue utilizada y las métricas están registradas. Reproducir un entrenamiento requeriría un nuevo conjunto de prueba sellado.  
