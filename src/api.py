import json
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src import evaluacion as ev
from src.preprocesamiento_api import ErrorValidacion, validar_y_preparar


def cargar_modelo_promovido():
    """Devuelve (pipeline, version) leyendo mejor_version.json; falla con mensaje claro si no existe."""
    if not ev.RUTA_MEJOR.exists():
        raise FileNotFoundError(f"No existe {ev.RUTA_MEJOR}. Ejecuta primero: python entrenar.py")
    version = json.loads(ev.RUTA_MEJOR.read_text(encoding="utf-8"))["version"]
    ruta = ev.DIR_MODELOS / f"{version}.joblib"
    if not ruta.exists():
        raise FileNotFoundError(f"Falta el modelo promovido: {ruta}")
    return joblib.load(ruta), version


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Carga el modelo promovido una sola vez, al arrancar."""
    app.state.modelo, app.state.version = cargar_modelo_promovido()
    yield


app = FastAPI(title="Alerta temprana de deserción", lifespan=ciclo_de_vida)


@app.middleware("http")
async def medir_latencia(request: Request, call_next):
    from time import perf_counter

    inicio = perf_counter()
    respuesta = await call_next(request)
    latencia_ms = (perf_counter() - inicio) * 1000

    respuesta.headers["X-Latencia-ms"] = f"{latencia_ms:.2f}"
    return respuesta


def _error_400(mensaje: str, campo: str) -> JSONResponse:
    return JSONResponse(status_code=400, content={"error": mensaje, "campo": campo})


@app.get("/salud")
def salud():
    """Devuelve el estado del servicio y la versión promovida cargada."""
    return {"estado": "ok", "version": app.state.version}


@app.post("/predecir")
async def predecir(request: Request):
    """Recibe los 8 campos en JSON; devuelve clase, probabilidades y versión (o 400 con el campo culpable)."""
    try:
        cuerpo = await request.json()
    except ValueError:
        return _error_400("El cuerpo no es un JSON válido.", "cuerpo")
    try:
        X = validar_y_preparar(cuerpo)
    except ErrorValidacion as e:
        return _error_400(e.mensaje, e.campo)

    modelo = app.state.modelo
    probs = modelo.predict_proba(X)[0]
    clases = [str(c) for c in modelo.classes_]
    return {
        "clase": clases[int(probs.argmax())],
        "probabilidades": {c: round(float(p), 4) for c, p in zip(clases, probs)},
        "version": app.state.version,
    }