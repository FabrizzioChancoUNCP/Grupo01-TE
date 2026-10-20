import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluacion as ev  # noqa: E402
from src.datos import CLASES, COLUMNA_OBJETIVO, VARIABLES_MODELO  # noqa: E402
from src.modelo import concluir_comparacion, concluir_con_variabilidad, definir_modelos  # noqa: E402


# ---------- ayudas ----------
def _ficha(version, f1, recall_dropout):
    """Ficha mínima falsa: solo lo que leen elegir_mejor_version y concluir_comparacion."""
    return {"version": version,
            "metricas_validacion": {"macro_f1": f1, "recall_por_clase": {"Dropout": recall_dropout}}}


def _ficha_completa(version, f1, recalls):
    """Ficha falsa con recall de las tres clases (para tabla_comparativa)."""
    return {"version": version,
            "metricas_validacion": {"macro_f1": f1, "recall_por_clase": dict(zip(CLASES, recalls))}}


def _df_sintetico(n=240, seed=0):
    """Datos falsos con las 8 variables; el Target depende de las unidades aprobadas (hay señal)."""
    rng = np.random.default_rng(seed)
    aprobadas = rng.integers(0, 7, n)
    df = pd.DataFrame({
        "Curricular units 1st sem (enrolled)": rng.integers(5, 8, n),
        "Curricular units 1st sem (approved)": aprobadas,
        "Curricular units 1st sem (grade)": rng.uniform(0, 20, n),
        "Tuition fees up to date": rng.integers(0, 2, n),
        "Debtor": rng.integers(0, 2, n),
        "Scholarship holder": rng.integers(0, 2, n),
        "Age at enrollment": rng.integers(18, 40, n),
        "Admission grade": rng.uniform(95, 190, n),
    })
    df[COLUMNA_OBJETIVO] = np.where(aprobadas <= 1, "Dropout", np.where(aprobadas <= 4, "Enrolled", "Graduate"))
    return df[VARIABLES_MODELO + [COLUMNA_OBJETIVO]]


@pytest.fixture
def rutas_temporales(tmp_path, monkeypatch):
    """Redirige todas las salidas a una carpeta temporal para no ensuciar modelos/ real."""
    monkeypatch.setattr(ev, "DIR_MODELOS", tmp_path / "modelos")
    monkeypatch.setattr(ev, "DIR_FICHAS", tmp_path / "modelos" / "fichas_json")
    monkeypatch.setattr(ev, "RUTA_MEJOR", tmp_path / "modelos" / "mejor_version.json")
    monkeypatch.setattr(ev, "RUTA_COMPARACION", tmp_path / "modelos" / "comparacion_validacion.csv")
    return tmp_path


# ---------- métricas ----------
def test_metricas_perfectas():
    y = ["Dropout", "Enrolled", "Graduate"]
    m = ev.calcular_metricas(y, y)
    assert m["macro_f1"] == 1.0 and m["recall_por_clase"]["Dropout"] == 1.0
    assert m["matriz_confusion"]["orden_clases"] == CLASES


def test_metricas_detectan_falso_negativo_de_dropout():
    real = ["Dropout", "Dropout", "Graduate", "Enrolled"]
    pred = ["Dropout", "Graduate", "Graduate", "Enrolled"]  # un Dropout no detectado
    m = ev.calcular_metricas(real, pred)
    assert m["recall_por_clase"]["Dropout"] == 0.5
    assert m["macro_f1"] < 1.0


# ---------- selección (solo validación) ----------
def test_elegir_mejor_por_macro_f1_y_con_piso():
    fichas = [_ficha("a", 0.60, 0.90), _ficha("b", 0.70, 0.50)]
    assert ev.elegir_mejor_version(fichas, None)["version"] == "b"
    assert ev.elegir_mejor_version(fichas, 0.80)["version"] == "a"
    with pytest.raises(ValueError):
        ev.elegir_mejor_version(fichas, 0.99)


# ---------- modelos ----------
def test_modelos_definidos_incluyen_base_simple_y_tres_mlp():
    nombres = set(definir_modelos())
    assert {"dummy", "regresion_logistica", "mlp_base", "mlp_profunda", "mlp_regularizada"} <= nombres


# ---------- conclusión honesta ----------
def test_conclusion_honesta_si_la_red_no_es_mejor():
    fichas = [_ficha("regresion_logistica", 0.72, 0.80), _ficha("mlp_base", 0.65, 0.70),
              _ficha("dummy", 0.20, 1.0), _ficha("mlp_profunda", 0.66, 0.71)]
    texto = concluir_comparacion(fichas)
    assert "NO supera" in texto and "regresion_logistica" in texto


def test_conclusion_cuando_la_red_si_es_mejor_y_cuando_empata():
    mejor = [_ficha("regresion_logistica", 0.60, 0.7), _ficha("mlp_base", 0.70, 0.8), _ficha("dummy", 0.2, 1.0)]
    assert "SÍ supera" in concluir_comparacion(mejor)
    empate = [_ficha("regresion_logistica", 0.700, 0.7), _ficha("mlp_base", 0.705, 0.8), _ficha("dummy", 0.2, 1.0)]
    assert "no se puede afirmar" in concluir_comparacion(empate)


# ---------- tabla comparativa ----------
def test_tabla_comparativa_ordena_de_mayor_a_menor(rutas_temporales):
    fichas = [_ficha_completa("a", 0.50, [0.6, 0.3, 0.7]),
              _ficha_completa("b", 0.70, [0.8, 0.4, 0.8]),
              _ficha_completa("c", 0.60, [0.7, 0.3, 0.7])]
    tabla = ev.tabla_comparativa(fichas)
    assert list(tabla["version"]) == ["b", "c", "a"]
    assert ev.RUTA_COMPARACION.exists()


# ---------- flujo completo ----------
def test_flujo_completo_fichas_promocion_y_prueba_una_sola_vez(rutas_temporales):
    df = _df_sintetico()
    train, val, prueba = df.iloc[:160], df.iloc[160:200], df.iloc[200:]

    mejor = ev.entrenar_y_promover(train, val, prueba)

    # una ficha por versión y exactamente una promovida
    fichas = [json.loads(p.read_text(encoding="utf-8")) for p in ev.DIR_FICHAS.glob("*.json")]
    assert len(fichas) == 5
    assert [f["promovida"] for f in fichas].count(True) == 1
    assert all(f["metricas_prueba"] is None for f in fichas)  # la prueba aún no se tocó
    assert ev.RUTA_MEJOR.exists() and mejor["version"] in {f["version"] for f in fichas}
    assert ev.RUTA_COMPARACION.exists()

    # cada ficha trae los campos que exige la consigna
    for f in fichas:
        assert {"version", "fecha", "datos", "parametros", "metricas_validacion", "metricas_prueba",
                "promovida"} <= set(f)
        assert f["datos"]["semilla"] == 42

    # la prueba se evalúa una vez y se registra en la ficha promovida
    metricas = ev.evaluar_en_prueba(prueba)
    assert 0.0 <= metricas["macro_f1"] <= 1.0
    ficha_promovida = json.loads(ev._ruta_ficha(mejor["version"]).read_text(encoding="utf-8"))
    assert ficha_promovida["metricas_prueba"] is not None

    # segundo intento: prohibido
    with pytest.raises(RuntimeError):
        ev.evaluar_en_prueba(prueba)

def test_no_se_reentrena_despues_de_ver_la_prueba(rutas_temporales):
    df = _df_sintetico()
    train, val, prueba = df.iloc[:160], df.iloc[160:200], df.iloc[200:]
    ev.entrenar_y_promover(train, val, prueba)
    ev.evaluar_en_prueba(prueba)
    with pytest.raises(RuntimeError):
        ev.entrenar_y_promover(train, val, prueba)


def test_conclusion_con_variabilidad_distingue_ruido_de_diferencia_real():
    cv = pd.DataFrame({"version": ["regresion_logistica", "mlp_base", "dummy"],
                       "macro_f1_cv_media": [0.66, 0.63, 0.22], "macro_f1_cv_std": [0.02, 0.02, 0.0]})
    assert "NO supera" in concluir_con_variabilidad(cv)
    cv.loc[cv["version"] == "mlp_base", "macro_f1_cv_media"] = 0.655
    assert "no se puede afirmar" in concluir_con_variabilidad(cv)


def test_medir_variabilidad_devuelve_media_y_desviacion_por_version(rutas_temporales, monkeypatch):
    monkeypatch.setattr(ev, "SEMILLAS_VARIABILIDAD", [42, 7])
    df = _df_sintetico()
    tabla = ev.medir_variabilidad(df[VARIABLES_MODELO], df[COLUMNA_OBJETIVO], n_splits=2)
    assert set(tabla["version"]) == set(definir_modelos())
    assert (tabla["n_evaluaciones"] == 4).all()
    assert (ev.DIR_MODELOS / "variabilidad_cv_train.csv").exists()


def test_auditar_por_grupo_marca_grupos_pequenos_y_exige_columnas():
    df = _df_sintetico()
    df["Gender"] = np.where(np.arange(len(df)) < 10, 1, 0)
    df["International"] = 0
    df["Educational special needs"] = 0
    modelo = definir_modelos()["regresion_logistica"].fit(df[VARIABLES_MODELO], df[COLUMNA_OBJETIVO])
    aud = ev.auditar_por_grupo(modelo, df)
    assert aud["Gender"]["1"]["n"] == 10 and aud["Gender"]["1"]["advertencia"] is not None
    assert aud["Gender"]["0"]["advertencia"] is None and "grupo_edad" in aud
    assert ev.auditar_por_grupo(modelo, df[VARIABLES_MODELO + [COLUMNA_OBJETIVO]]) == {}


def test_regla_decision_recall_dropout_no_baja_al_subir_el_peso(rutas_temporales):
    df = _df_sintetico()
    X, y = df[VARIABLES_MODELO], df[COLUMNA_OBJETIVO]
    modelo = definir_modelos()["regresion_logistica"].fit(X.iloc[:160], y.iloc[:160])
    tabla = ev.analizar_regla_decision(modelo, X.iloc[160:], y.iloc[160:])
    assert tabla["recall_Dropout"].is_monotonic_increasing
    base = ev.calcular_metricas(y.iloc[160:], modelo.predict(X.iloc[160:]))
    assert tabla.loc[tabla["peso_dropout"] == 1.0, "macro_f1_val"].iloc[0] == base["macro_f1"]


def test_comparar_sin_variables_solo_cv_sobre_train(monkeypatch):
    monkeypatch.setattr(ev, "SEMILLAS_VARIABILIDAD", [42])
    tabla = ev.comparar_sin_variables(_df_sintetico(), {"sin_edad": ["Age at enrollment"]}, n_splits=3)
    assert list(tabla["escenario"]) == ["con_todas", "sin_edad"]
    assert list(tabla["n_variables"]) == [8, 7]

def test_comparar_sin_variables_conserva_grupos(monkeypatch):
    df = _df_sintetico(n=90)
    df.loc[1, VARIABLES_MODELO] = df.loc[0, VARIABLES_MODELO]
    df.loc[1, "Age at enrollment"] += 1

    grupos_registrados = []

    def cv_simulada(modelo, X, y, *, cv, groups, scoring):
        grupos_registrados.append(groups.to_numpy().copy())
        return np.full(cv.n_splits, 0.5)

    monkeypatch.setattr(ev, "SEMILLAS_VARIABILIDAD", [42])
    monkeypatch.setattr(ev, "cross_val_score", cv_simulada)

    ev.comparar_sin_variables(
        df, {"sin_edad": ["Age at enrollment"]}, n_splits=2
    )

    np.testing.assert_array_equal(
        grupos_registrados[0], grupos_registrados[1]
    )


def test_pipelines_reales_escalan_solo_con_train():
    df = _df_sintetico()
    X, y = df[VARIABLES_MODELO], df[COLUMNA_OBJETIVO]
    for nombre, pipe in definir_modelos().items():
        pipe.fit(X.iloc[:160], y.iloc[:160])
        esc = pipe.named_steps["escalador"]
        assert esc.mean_ == pytest.approx(X.iloc[:160].mean().to_numpy())
        assert not np.allclose(esc.mean_, X.mean().to_numpy())