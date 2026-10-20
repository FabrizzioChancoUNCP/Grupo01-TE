# proyecto_grupo01
Trabajo final de Tecnologias Emergente


## Etapa E5: API de predicción

Se incorporaron `src/api.py`, `src/preprocesamiento_api.py` y `tests/test_api.py` de la etapa E5 sobre la base E3-E4-v2. Se conservaron los modelos entrenados, la evaluación, los resultados y el resto de las pruebas de E3-E4-v2.

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
python -m pytest -q
uvicorn src.api:app --reload
```

La API expone `GET /salud`, `POST /predecir` y documentación interactiva en `/docs`. Usa automáticamente la versión indicada en `modelos/mejor_version.json`; no es necesario reentrenar.

Ejemplo de JSON para `POST /predecir`:

```json
{
  "unidades_inscritas_s1": 6,
  "unidades_aprobadas_s1": 5,
  "promedio_s1": 12.5,
  "matricula_al_dia": 1,
  "deudor": 0,
  "becario": 1,
  "edad_ingreso": 19,
  "nota_admision": 130.0
}
```

Las entradas inválidas devuelven HTTP 400 con un mensaje y el campo responsable.

## Etapa E6: chatbot de reglas

El chatbot E6 quedó integrado con la API E5 y los modelos y resultados existentes de E3-E4-v2. Desde la raíz del proyecto, en dos terminales:

```bash
python -m pip install -r requirements.txt
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

```bash
python -m src.chatbot
```

El chatbot solicita **8 datos** de forma secuencial, valida números, rangos y respuestas sí/no, permite **3 intentos por pregunta** y sugiere acudir al tutor si identifica `Dropout`, si no comprende las respuestas o si hay una falla del servicio. No registra una cita ni notifica al tutor. Por defecto no hay umbral probabilístico configurado: no se debe fijar sin validación sobre el conjunto de validación.

Pruebas:

```bash
python -m pytest tests/test_chatbot.py tests/test_api.py tests/test_datos.py -q
python -m pytest -q
```

## Estado de verificación (18/10/2026)

- **Funciona** el recorrido chatbot -> API -> modelo promovido (`regresion_logistica`) con un ejemplo real; `/salud` responde 200, `/predecir` responde 200 con tres probabilidades y una entrada inválida devuelve 400.
- Suite completa en este entorno (Python 3.13, scikit-learn 1.8.0): **71 pruebas exitosas y 4 fallos**. Los fallos corresponden a `MLPClassifier(early_stopping=True)` con clases de texto y una incompatibilidad `np.isnan` en la implementación de sklearn 1.8.0, no al chatbot ni a la API. Por tanto el reentrenamiento integral NO queda validado.
- Los archivos `.joblib` guardados indican que se generaron con scikit-learn 1.9.1 y producen `InconsistentVersionWarning` al cargarlos en 1.8.0. Aunque una predicción de prueba funciona, **no se garantiza compatibilidad del modelo** en este entorno. Para una entrega reproducible, ejecutar en un entorno con la misma versión usada para entrenar, fijar dependencias y volver a ejecutar todos los tests; verificar además el modelo con muestras de referencia.
- El contexto aún contiene campos `[COMPLETAR]` que debe resolver el grupo; **`informe.pdf` está incluido (7 páginas)**, y el registro de IA y revisión por PR deben completarse por integrantes reales. Se conservan los artefactos de E3-E4 y las decisiones de modelado; no se reentrenó ni recalculó la prueba final.

## Corrección de reentrenamiento MLP y cierre E5 (18/10/2026)

- Se resolvieron los 4 fallos de pruebas de MLP con `scikit-learn==1.8.0`: parada por convergencia de pérdida de entrenamiento en lugar del modo `early_stopping=True` incompatible con etiquetas de texto de este proyecto.
- `python -m pytest -q` — **76/76 pruebas aprobadas** en Python 3.13 y dependencias fijadas en `requirements.txt`.
- **Importante:** el modelo promovido ya tiene métricas de prueba registradas. **No ejecutar `python entrenar.py` ni `python entrenar.py --final` sobre estos artefactos para repetir la selección o evaluación.** Los modelos entregados se conservan tal cual.
- Los artefactos `.joblib` históricos fueron generados con scikit-learn 1.9.1; la versión 1.8.0 usada para verificar el código genera advertencias al cargarlos. No publicar el servicio antes de resolver esa discrepancia de entorno.
- Ver el contrato completo, las instrucciones y los acuerdos del grupo pendientes en **`docs/ETAPA_E5.md`**. Se anotaron implementaciones verificadas en `CONTEXTO.md`, dejando explícita su necesidad de ratificación.


## Reproducción desde cero y trazabilidad (revisión 18/10/2026)

1. Usar **Python 3.13**, crear un entorno virtual (`python -m venv .venv`) y activarlo; instalar las dependencias con `python -m pip install -r requirements.txt`.
2. El archivo **`data/data.csv` ya viene incluido** en esta entrega ZIP. Si no está presente en un clon, obtener el conjunto **Predict Students’ Dropout and Academic Success** desde [UCI, dataset 697](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success), colocar el CSV original en `data/data.csv` y comprobar su formato: 4424 registros y separador `;`.
3. Conservar `modelos/*.joblib`, `modelos/fichas_json/*.json` y `modelos/mejor_version.json` para **reproducir la inferencia del modelo ya promovido** sin reentrenar. Ejecutar `python -m pytest -q` y después iniciar la API y el chatbot con los comandos anteriores.
4. `notebooks/01_exploracion.ipynb` y `notebooks/02_resultados.ipynb` conservan resultados ejecutados; los archivos CSV de `modelos/` y `docs/resumen_eda.json` son referencias de los valores históricos.
5. **No volver a ejecutar `entrenar.py` sobre la selección entregada:** la prueba final ya fue utilizada una vez y el script protege contra repetir la evaluación. Reproducir un entrenamiento experimental requeriría un protocolo independiente con un nuevo conjunto de prueba sellado; sus números no reemplazan los de la entrega.

### Limitación reproducible pendiente

Las fichas y los modelos MLP históricos (`mlp_profunda` y `mlp_regularizada`) usan `early_stopping=True` con un 10 % de validación interna, pero `src/modelo.py` actualmente usa `early_stopping=False` para permitir la ejecución de pruebas con scikit-learn 1.8.0. **No es correcto afirmar que los resultados históricos se regeneran exactamente con el código actual.** Además, los `.joblib` señalan scikit-learn 1.9.1 frente a 1.8.0 en `requirements.txt`; se observó `InconsistentVersionWarning`. Antes de una reproducción exacta o despliegue se necesita disponer de un entorno compatible, resolver y verificar la configuración original, y repetir las pruebas sin reutilizar el conjunto de prueba para seleccionar modelos. No se han alterado fichas ni métricas históricas para ocultar esta diferencia.

### Verificaciones de entrega pendientes del grupo

Confirmar el nombre y contribuciones reales de cada integrante, completar `docs/uso_ia.md` con autorías reales, comprobar commits/PR en GitHub y acceso docente; revisar limitaciones éticas y compatibilidad antes de producción. Para verificar la API con latencia, ejecutar `python scripts/medir_latencia_api.py` (servicio levantado); reportar los valores únicamente después de medirlos en el equipo definitivo.


## Verificación adicional y alcance (18/10/2026)

- La cifra **0.6514 corresponde a macro-F1 de validación**, no al conjunto de prueba. La ficha `modelos/fichas_json/regresion_logistica.json` registra **0.6537 de macro-F1 de prueba** (redondeado a cuatro decimales).
- Las fichas están en `modelos/fichas_json/<version>.json`, no directamente en `modelos/<version>.json`. La decisión promovida figura en `modelos/mejor_version.json`.
- `entrenar.py` admite `--variabilidad` y `--final`; **no deben usarse para repetir la selección o reevaluar el test sellado de esta entrega**. Consultar `docs/REPRODUCIBILIDAD.md` para comprobaciones de solo lectura y límites del reentrenamiento.
- En el chatbot puede solicitarse `hablar con un tutor` durante cualquier pregunta. Esta acción no agenda ni transmite datos personales.
- Se conservó la regla de decisión sin umbral/ponderación nueva: las alternativas de peso en validación mejoran recall de Dropout a costa de macro-F1 y recall de otras clases. No se modifica el modelo sin aprobación humana ni nueva validación.
- `docs/AUDITORIA_CORRECCIONES.md` separa evidencias verificadas, cambios y requisitos pendientes. La entrega no incluye historial `.git` verificable; los commits y PR deben comprobarse en el repositorio remoto auténtico.
