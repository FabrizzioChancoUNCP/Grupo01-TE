import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.datos import (  # noqa: E402
    CLASES,
    COLUMNAS_AUDITORIA,
    COLUMNA_OBJETIVO,
    COLUMNAS_FUGA,
    COLUMNAS_SENSIBLES,
    MAPA_API_A_COLUMNA,
    RANGOS_DOMINIO,
    RUTA_DATOS,
    SEED,
    VARIABLES_MODELO,
    cargar_datos,
    cargar_y_particionar,
    contar_duplicados_conflictivos,
    diagnosticar_calidad,
    eliminar_duplicados,
    limpiar,
    obtener_grupos,
    particionar,
    resumen_particion,
    separar_xy,
    verificar_sin_fuga,
)


def _df_sintetico() -> pd.DataFrame:
    objetivo = ["Graduate"] * 100 + ["Dropout"] * 60 + ["Enrolled"] * 40
    return pd.DataFrame({"x": range(200), "Target": objetivo})


def test_particion_tamanos_estratos_y_sin_solapes():
    df = _df_sintetico()
    train, val, prueba = particionar(df)
    assert len(train) + len(val) + len(prueba) == len(df)
    assert abs(len(train) - 140) <= 2 and abs(len(val) - 30) <= 2 and abs(len(prueba) - 30) <= 2
    assert not (set(train.index) & set(val.index) or set(train.index) & set(prueba.index)
                or set(val.index) & set(prueba.index))
    for parte in (train, val, prueba):  # misma proporción de clases que el total
        assert parte["Target"].value_counts(normalize=True)["Enrolled"] == pytest.approx(0.20, abs=0.03)


def test_particion_reproducible_con_semilla():
    a = particionar(_df_sintetico(), seed=SEED)
    b = particionar(_df_sintetico(), seed=SEED)
    assert all(x.index.equals(y.index) for x, y in zip(a, b))


def test_eliminar_duplicados():
    df = pd.concat([_df_sintetico(), _df_sintetico().iloc[:5]], ignore_index=True)
    assert len(eliminar_duplicados(df)) == 200


def test_error_sin_columna_objetivo():
    with pytest.raises(KeyError):
        particionar(pd.DataFrame({"x": [1, 2, 3]}))


def test_error_archivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        cargar_datos(tmp_path / "no_existe.csv")

def test_diagnostico_calidad_conteos_basicos():
    df = pd.DataFrame({
        "Curricular units 1st sem (enrolled)": [0, 6, 0, 6],
        "Curricular units 1st sem (approved)": [0, 5, 0, 5],
        "Curricular units 1st sem (grade)": [0.0, 12.0, 0.0, 12.0],
        "Target": ["Dropout", "Graduate", "Dropout", "Graduate"],
    })
    d = diagnosticar_calidad(df)
    assert d["filas"] == 4
    assert d["nulos_por_columna"] == {}
    assert d["duplicados_exactos"] == 2  # filas 2 y 4 repiten a las 1 y 3
    assert d["sin_unidades_inscritas_s1"] == 2
    assert d["clases"] == {"Dropout": 2, "Graduate": 2}


def test_diagnostico_error_sin_objetivo():
    with pytest.raises(KeyError):
        diagnosticar_calidad(pd.DataFrame({"Curricular units 1st sem (enrolled)": [1]}))

def test_cargar_quita_tabuladores(tmp_path):
    ruta = tmp_path / "d.csv"
    ruta.write_text("Daytime/evening attendance\t;Target\n1;Dropout\n", encoding="utf-8")
    assert list(cargar_datos(ruta).columns) == ["Daytime/evening attendance", "Target"]


def test_limpiar_quita_espacios():
    df = pd.DataFrame({"x ": [1, 2], "Target ": [" Dropout", "Graduate\t"]})
    r = limpiar(df)
    assert list(r.columns) == ["x", "Target"] and list(r["Target"]) == ["Dropout", "Graduate"]


def test_limpiar_rechaza_clase_desconocida():
    with pytest.raises(ValueError):
        limpiar(pd.DataFrame({"Target": ["Dropuot", "Graduate"]}))


def test_duplicados_se_quitan_antes_de_particionar():
    df = pd.concat([_df_sintetico(), _df_sintetico().iloc[:5]], ignore_index=True)
    train, val, prueba = particionar(eliminar_duplicados(df))
    verificar_sin_fuga(train, val, prueba)  # no debe lanzar
    assert len(train) + len(val) + len(prueba) == 200


def test_verificar_sin_fuga_detecta_filas_e_indices_repetidos():
    df = _df_sintetico()
    clon = df.iloc[:10].copy()
    clon.index = range(100, 110)  # mismas filas, distinto índice
    with pytest.raises(ValueError, match="idénticas"):
        verificar_sin_fuga(df.iloc[:10], df.iloc[10:20], clon)
    with pytest.raises(ValueError, match="índices"):
        verificar_sin_fuga(df.iloc[:10], df.iloc[10:20], df.iloc[5:15])


def test_duplicados_conflictivos():
    conflicto = pd.DataFrame({"x": [1, 1, 2, 3], "Target": ["Dropout", "Graduate", "Dropout", "Graduate"]})
    assert contar_duplicados_conflictivos(conflicto) == 1
    assert contar_duplicados_conflictivos(_df_sintetico()) == 0


def test_variables_modelo_sin_fuga_ni_sensibles():
    assert len(VARIABLES_MODELO) == 8
    assert not set(VARIABLES_MODELO) & (set(COLUMNAS_FUGA) | set(COLUMNAS_SENSIBLES) | {COLUMNA_OBJETIVO})
    assert set(MAPA_API_A_COLUMNA.values()) == set(VARIABLES_MODELO)
    assert set(RANGOS_DOMINIO) == set(VARIABLES_MODELO)


def test_separar_xy_rechaza_variable_prohibida_y_acepta_las_permitidas():
    df = pd.DataFrame({v: [1, 2] for v in VARIABLES_MODELO})
    df[COLUMNAS_FUGA[0]] = [1, 2]
    df["Gender"] = [0, 1]
    df[COLUMNA_OBJETIVO] = ["Dropout", "Graduate"]
    X, y = separar_xy(df)
    assert list(X.columns) == VARIABLES_MODELO and len(y) == 2
    with pytest.raises(ValueError):
        separar_xy(df, [COLUMNAS_FUGA[0]])
    with pytest.raises(ValueError):
        separar_xy(df, ["Gender"])
    with pytest.raises(KeyError):
        separar_xy(df.drop(columns=["Debtor"]))


def test_obtener_grupos():
    df = pd.DataFrame({c: [0, 1, 0] for c in COLUMNAS_AUDITORIA})
    df["Age at enrollment"] = [18, 22, 30]
    g = obtener_grupos(df)
    assert list(g["grupo_edad"]) == ["<=20", "21-25", ">25"]
    assert "Age at enrollment" not in g.columns


def test_diagnostico_incluye_accuracy_clase_mayoritaria():
    df = pd.DataFrame({
        "Curricular units 1st sem (enrolled)": [6, 6, 6],
        "Curricular units 1st sem (approved)": [5, 5, 5],
        "Curricular units 1st sem (grade)": [12.0, 12.0, 12.0],
        "Target": ["Graduate", "Graduate", "Dropout"],
    })
    assert diagnosticar_calidad(df)["accuracy_si_siempre_clase_mayoritaria"] == pytest.approx(0.6667, abs=1e-4)


def test_escalador_ajustado_solo_con_train():
    X_train, X_val = pd.DataFrame({"a": [0.0, 10.0]}), pd.DataFrame({"a": [100.0, 200.0]})
    esc = StandardScaler().fit(X_train)             # fit SOLO con entrenamiento
    media_antes = esc.mean_.copy()
    Xv = esc.transform(X_val)                       # validación solo se transforma
    assert esc.mean_[0] == pytest.approx(5.0)       # media de train, no de val
    assert np.array_equal(esc.mean_, media_antes)   # transform no reajusta
    assert Xv[0, 0] == pytest.approx(19.0)          # (100-5)/5


@pytest.mark.skipif(not RUTA_DATOS.exists(), reason="data/data.csv no está")
def test_datos_reales_cumplen_el_contrato():
    train, val, prueba = cargar_y_particionar()
    total = len(train) + len(val) + len(prueba)
    assert abs(len(train) / total - 0.70) < 0.01
    assert abs(len(val) / total - 0.15) < 0.01
    assert abs(len(prueba) / total - 0.15) < 0.01
    for parte in (train, val, prueba):
        assert set(parte[COLUMNA_OBJETIVO]) <= set(CLASES)
    verificar_sin_fuga(train, val, prueba)
    X, _ = separar_xy(train)
    assert list(X.columns) == VARIABLES_MODELO
    # Si alguna de las dos falla, es un hallazgo: documenta la decisión en 2.3 antes de relajar la prueba.
    assert X.isna().sum().sum() == 0
    for col, (minimo, maximo) in RANGOS_DOMINIO.items():
        assert X[col].between(minimo, maximo).all(), col

def test_filas_compartidas_solo_8_variables():
    import pandas as pd
    from src.datos import VARIABLES_MODELO, contar_filas_modelo_compartidas
    base = {col: [1, 2] for col in VARIABLES_MODELO}
    a = pd.DataFrame(base)
    b = pd.DataFrame({col: [1, 3] for col in VARIABLES_MODELO})
    assert contar_filas_modelo_compartidas(a, b) == 1
    assert contar_filas_modelo_compartidas(a, a) == 2


def test_resumen_eda_contiene_cifras_y_grupos():
    from src.datos import cargar_datos, limpiar, eliminar_duplicados, resumen_eda, RUTA_DATOS
    reporte = resumen_eda(eliminar_duplicados(limpiar(cargar_datos(RUTA_DATOS))))
    assert "perfiles_conflictivos_8_variables" in reporte
    assert "clases_sin_unidades_s1" in reporte
    assert "tasa_dropout_por_variable" in reporte
    assert "Scholarship holder" in reporte["tasa_dropout_por_variable"]
