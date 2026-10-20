import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import api as api_mod  # noqa: E402
from src import evaluacion as ev  # noqa: E402
from src.datos import CLASES, VARIABLES_MODELO  # noqa: E402

PAYLOAD_OK = {
    "unidades_inscritas_s1": 6, "unidades_aprobadas_s1": 5, "promedio_s1": 12.5,
    "matricula_al_dia": 1, "deudor": 0, "becario": 1, "edad_ingreso": 19, "nota_admision": 130.0,
}


def _pipeline_falso():
    """Modelo mínimo con las 8 columnas reales y las 3 clases (no usa los datos del proyecto)."""
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame({c: rng.uniform(0, 10, n) for c in VARIABLES_MODELO})
    y = np.array(CLASES * (n // 3))
    return Pipeline([("escalador", StandardScaler()), ("clasificador", LogisticRegression(max_iter=500))]).fit(X, y)


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    """Cliente con un modelo 'promovido' falso en carpeta temporal (lanza el arranque de la app)."""
    joblib.dump(_pipeline_falso(), tmp_path / "vtest.joblib")
    (tmp_path / "mejor_version.json").write_text(json.dumps({"version": "vtest"}), encoding="utf-8")
    monkeypatch.setattr(ev, "DIR_MODELOS", tmp_path)
    monkeypatch.setattr(ev, "RUTA_MEJOR", tmp_path / "mejor_version.json")
    with TestClient(api_mod.app) as c:
        yield c


# ---------- éxito ----------
def test_salud(cliente):
    r = cliente.get("/salud")
    assert r.status_code == 200
    assert r.json() == {"estado": "ok", "version": "vtest"}


def test_predecir_exitoso(cliente):
    r = cliente.post("/predecir", json=PAYLOAD_OK)
    assert r.status_code == 200
    cuerpo = r.json()
    assert set(cuerpo) == {"clase", "probabilidades", "version"}
    assert cuerpo["clase"] in CLASES and cuerpo["version"] == "vtest"
    assert set(cuerpo["probabilidades"]) == set(CLASES)
    assert sum(cuerpo["probabilidades"].values()) == pytest.approx(1.0, abs=1e-3)


def test_cabecera_latencia_en_respuestas(cliente):
    respuestas = [
        cliente.get("/salud"),
        cliente.post("/predecir", json=PAYLOAD_OK),
        cliente.post("/predecir", json={}),
    ]

    assert [r.status_code for r in respuestas] == [200, 200, 400]

    for respuesta in respuestas:
        latencia = respuesta.headers.get("X-Latencia-ms")
        assert latencia is not None
        assert float(latencia) >= 0


# ---------- errores 400 ----------
def _post_con(cliente, **cambios):
    return cliente.post("/predecir", json={**PAYLOAD_OK, **cambios})


def _assert_400(r, campo):
    assert r.status_code == 400
    cuerpo = r.json()
    assert set(cuerpo) == {"error", "campo"} and cuerpo["campo"] == campo and cuerpo["error"]


def test_campo_no_permitido(cliente):
    _assert_400(_post_con(cliente, nombre="Ana"), "nombre")


def test_numero_extremadamente_grande(cliente):
    respuesta = _post_con(cliente, nota_admision=10**400)
    _assert_400(respuesta, "nota_admision")


@pytest.mark.parametrize("campo", list(PAYLOAD_OK))
def test_campo_faltante(cliente, campo):
    datos = {k: v for k, v in PAYLOAD_OK.items() if k != campo}
    _assert_400(cliente.post("/predecir", json=datos), campo)


@pytest.mark.parametrize("campo,valor", [
    ("unidades_inscritas_s1", 61), ("unidades_aprobadas_s1", -1), ("promedio_s1", 20.5),
    ("matricula_al_dia", 2), ("deudor", -1), ("becario", 5),
    ("edad_ingreso", 15), ("edad_ingreso", 81), ("nota_admision", 200.1),
])
def test_fuera_de_rango(cliente, campo, valor):
    _assert_400(_post_con(cliente, **{campo: valor}), campo)


@pytest.mark.parametrize("campo,valor", [
    ("promedio_s1", "doce"), ("deudor", True), ("edad_ingreso", 19.5), ("becario", None), ("nota_admision", [130]),
])
def test_tipo_invalido(cliente, campo, valor):
    _assert_400(_post_con(cliente, **{campo: valor}), campo)


def test_aprobadas_mayor_que_inscritas(cliente):
    _assert_400(_post_con(cliente, unidades_inscritas_s1=4, unidades_aprobadas_s1=5), "unidades_aprobadas_s1")


def test_json_invalido_y_cuerpo_no_objeto(cliente):
    r = cliente.post("/predecir", content="no es json", headers={"Content-Type": "application/json"})
    _assert_400(r, "cuerpo")
    _assert_400(cliente.post("/predecir", json=[1, 2, 3]), "cuerpo")


# ---------- arranque ----------
def test_arranque_falla_si_no_hay_modelo_promovido(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "DIR_MODELOS", tmp_path)
    monkeypatch.setattr(ev, "RUTA_MEJOR", tmp_path / "no_existe.json")
    with pytest.raises(FileNotFoundError):
        with TestClient(api_mod.app):
            pass