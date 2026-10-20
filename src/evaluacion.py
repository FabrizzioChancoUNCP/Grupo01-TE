# Metricas validadas por Yuan-S29
# src/evaluacion.py
"""Etapa 4: evaluación, versionado y promoción.

Reglas: la versión ganadora se elige SOLO con validación; la prueba se evalúa UNA vez, al final.
"""
import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.datos import CLASES, COLUMNAS_AUDITORIA, SEED, obtener_grupos, separar_xy
from src.modelo import definir_modelos, describir_modelo, concluir_comparacion
from src.datos import VARIABLES_MODELO

RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / "modelos"
DIR_FICHAS = DIR_MODELOS / "fichas_json"          # contrato original: modelos/<version>.json
RUTA_MEJOR = DIR_MODELOS / "mejor_version.json"
MIN_GRUPO = 30
# Piso de recall de Dropout: lo define el GRUPO con resultados de validación (None = sin piso).
PISO_RECALL_DROPOUT = None

SEMILLAS_VARIABILIDAD = [42, 7, 123, 2024, 99]
# ---------- utilidades de lectura/escritura ----------
def _escribir_json(ruta: Path, contenido: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, indent=2, ensure_ascii=False), encoding="utf-8")


def _leer_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def _ruta_ficha(version: str) -> Path:
    return DIR_FICHAS / f"{version}.json"


def _ruta_modelo(version: str) -> Path:
    return DIR_MODELOS / f"{version}.joblib"


# ---------- métricas ----------
def calcular_metricas(y_real, y_pred) -> dict:
    """Recibe etiquetas reales y predichas; devuelve macro-F1, recall por clase y matriz de confusión."""
    recalls = recall_score(y_real, y_pred, labels=CLASES, average=None, zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y_real, y_pred, labels=CLASES, average="macro", zero_division=0)), 4),
        "recall_por_clase": {c: round(float(r), 4) for c, r in zip(CLASES, recalls)},
        "matriz_confusion": {
            "orden_clases": CLASES,  # filas = real, columnas = predicho
            "valores": confusion_matrix(y_real, y_pred, labels=CLASES).tolist(),
        },
    }

def auditar_por_grupo(pipeline, df: pd.DataFrame) -> dict:
    """Recibe un modelo entrenado y un DataFrame completo; devuelve n, recall de Dropout y macro-F1 por grupo."""
    if not set(COLUMNAS_AUDITORIA) <= set(df.columns):
        return {}
    X, y = separar_xy(df)
    pred = pd.Series(pipeline.predict(X), index=df.index)
    grupos = obtener_grupos(df)
    salida = {}
    for col in grupos.columns:
        salida[col] = {}
        for valor, idx in grupos.groupby(col).groups.items():
            y_g, p_g = y.loc[idx], pred.loc[idx]
            salida[col][str(valor)] = {
                "n": int(len(idx)),
                "recall_dropout": (
                    round(float(recall_score(y_g, p_g, labels=["Dropout"], average=None, zero_division=0)[0]), 4)
                    if (y_g == "Dropout").any() else None
                ),
                "macro_f1": round(float(f1_score(y_g, p_g, labels=sorted(set(y_g) | set(p_g)), average="macro", zero_division=0)), 4),
                "advertencia": f"n<{MIN_GRUPO}: interpretar con cautela" if len(idx) < MIN_GRUPO else None,
            }
    return salida

# ---------- entrenamiento + ficha por versión ----------
def entrenar_y_evaluar_version(version, pipeline, X_train, y_train, X_val, y_val, filas: dict) -> dict:
    """Entrena con TRAIN, evalúa con VALIDACIÓN, guarda modelo (.joblib) y ficha JSON. Devuelve la ficha.

    'filas' = {'filas_train':..., 'filas_val':..., 'filas_test':...}. No toca la prueba.
    """
    pipeline.fit(X_train, y_train)  # el escalador y el clasificador se ajustan solo con train
    metricas_val = calcular_metricas(y_val, pipeline.predict(X_val))

    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, _ruta_modelo(version))

    ficha = {
        "version": version,
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "datos": {"archivo": "data.csv", **filas, "semilla": SEED},
        "parametros": describir_modelo(pipeline),
        "metricas_validacion": metricas_val,
        "metricas_prueba": None,   # se llena una sola vez, solo para la versión promovida
        "promovida": False,
        "variables": VARIABLES_MODELO,
    }
    _escribir_json(_ruta_ficha(version), ficha)
    return ficha


# ---------- selección y promoción (solo validación) ----------
def elegir_mejor_version(fichas: list, piso_recall_dropout=PISO_RECALL_DROPOUT) -> dict:
    """Devuelve la ficha ganadora: mayor macro-F1 en validación (entre las que cumplan el piso, si existe)."""
    candidatas = fichas
    if piso_recall_dropout is not None:
        candidatas = [f for f in fichas
                      if f["metricas_validacion"]["recall_por_clase"]["Dropout"] >= piso_recall_dropout]
        if not candidatas:
            raise ValueError(f"Ninguna versión alcanza recall de Dropout >= {piso_recall_dropout} en validación")
    return max(candidatas, key=lambda f: f["metricas_validacion"]["macro_f1"])


def promover_version(ganadora: dict, fichas: list, piso_recall_dropout=PISO_RECALL_DROPOUT) -> dict:
    """Marca la ganadora como promovida en su ficha y escribe mejor_version.json."""
    for f in fichas:
        f["promovida"] = f["version"] == ganadora["version"]
        _escribir_json(_ruta_ficha(f["version"]), f)
    mejor = {
        "version": ganadora["version"],
        "macro_f1_validacion": ganadora["metricas_validacion"]["macro_f1"],
        "recall_dropout_validacion": ganadora["metricas_validacion"]["recall_por_clase"]["Dropout"],
        "criterio": "mayor macro-F1 en validación"
                    + (f" con recall de Dropout >= {piso_recall_dropout}" if piso_recall_dropout is not None else ""),
        "fecha_promocion": datetime.now().isoformat(timespec="seconds"),
    }
    _escribir_json(RUTA_MEJOR, mejor)
    return mejor

def _verificar_prueba_no_vista() -> None:
    """Lanza RuntimeError si alguna ficha ya tiene metricas_prueba: reentrenar tras ver la prueba la invalida."""
    for ruta in DIR_FICHAS.glob("*.json"):
        if _leer_json(ruta).get("metricas_prueba") is not None:
            raise RuntimeError("La prueba ya se evaluó. No se reentrena ni se vuelve a seleccionar.")

def entrenar_y_promover(train: pd.DataFrame, val: pd.DataFrame, prueba: pd.DataFrame,
                        piso_recall_dropout=PISO_RECALL_DROPOUT) -> dict:
    """Entrena todas las versiones, escribe fichas y promueve la mejor según VALIDACIÓN. No evalúa la prueba."""
    _verificar_prueba_no_vista()
    from src.datos import separar_xy  # import local para no acoplar el módulo al cargar
    X_train, y_train = separar_xy(train)
    X_val, y_val = separar_xy(val)
    filas = {"filas_train": len(train), "filas_val": len(val), "filas_test": len(prueba)}

    fichas = []
    for version, pipeline in definir_modelos().items():
        ficha = entrenar_y_evaluar_version(version, pipeline, X_train, y_train, X_val, y_val, filas)
        print(f"{version:22s} macro-F1 val = {ficha['metricas_validacion']['macro_f1']:.4f} | "
              f"recall Dropout = {ficha['metricas_validacion']['recall_por_clase']['Dropout']:.4f}")
        fichas.append(ficha)
        
    print("\n" + tabla_comparativa(fichas).to_string(index=False) + "\n")
    print(concluir_comparacion(fichas) + "\n")
    ganadora = elegir_mejor_version(fichas, piso_recall_dropout)
    mejor = promover_version(ganadora, fichas, piso_recall_dropout)
    ficha_g = _leer_json(_ruta_ficha(mejor["version"]))
    ficha_g["auditoria_validacion"] = auditar_por_grupo(joblib.load(_ruta_modelo(mejor["version"])), val)
    _escribir_json(_ruta_ficha(mejor["version"]), ficha_g)
    return mejor


# ---------- prueba: UNA sola vez, con la versión promovida ----------
def evaluar_en_prueba(prueba: pd.DataFrame) -> dict:
    """Evalúa la versión promovida sobre la prueba, una única vez, y guarda el resultado en su ficha.

    Lanza RuntimeError si la ficha ya tiene metricas_prueba (prohibido volver a mirar la prueba).
    """
    from src.datos import separar_xy
    version = _leer_json(RUTA_MEJOR)["version"]
    ficha = _leer_json(_ruta_ficha(version))
    if ficha["metricas_prueba"] is not None:
        raise RuntimeError(f"La prueba ya se evaluó para '{version}'. No se repite ni se usa para decidir.")

    filas_esperadas = ficha["datos"]["filas_test"]
    if len(prueba) != filas_esperadas:
        raise RuntimeError(
            f"La prueba actual tiene {len(prueba)} filas, pero el modelo "
            f"'{version}' fue entrenado con una partición que registraba "
            f"{filas_esperadas} filas de prueba."
        )

    X_prueba, y_prueba = separar_xy(prueba)
    modelo = joblib.load(_ruta_modelo(version))
    ficha["metricas_prueba"] = calcular_metricas(y_prueba, modelo.predict(X_prueba))
    ficha["auditoria_prueba"] = auditar_por_grupo(modelo, prueba)
    _escribir_json(_ruta_ficha(version), ficha)
    return ficha["metricas_prueba"]

RUTA_COMPARACION = DIR_MODELOS / "comparacion_validacion.csv"


def tabla_comparativa(fichas: list) -> pd.DataFrame:
    """Devuelve (y guarda en CSV) una tabla ordenada de versiones por macro-F1 de validación."""
    filas = []
    for f in fichas:
        m = f["metricas_validacion"]
        filas.append({
            "version": f["version"],
            "macro_f1_val": m["macro_f1"],
            **{f"recall_{c}": m["recall_por_clase"][c] for c in CLASES},
        })
    tabla = pd.DataFrame(filas).sort_values("macro_f1_val", ascending=False).reset_index(drop=True)
    RUTA_COMPARACION.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_COMPARACION, index=False)
    return tabla

def medir_variabilidad(X_train, y_train, n_splits: int = 5) -> pd.DataFrame:
    """CV estratificada con varias semillas SOLO sobre train; devuelve media y desviación del macro-F1 por versión."""
    filas = []
    for version, pipeline in definir_modelos().items():
        puntajes = []
        for semilla in SEMILLAS_VARIABILIDAD:
            modelo = clone(pipeline)
            if "random_state" in modelo.named_steps["clasificador"].get_params():
                modelo.set_params(clasificador__random_state=semilla)
            from sklearn.model_selection import StratifiedGroupKFold

            grupos = pd.util.hash_pandas_object(X_train, index=False)
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=semilla
            )
            puntajes.extend(cross_val_score(
                modelo, X_train, y_train,
                cv=cv,
                groups=grupos,
                scoring="f1_macro"
            ))
        filas.append({"version": version, "macro_f1_cv_media": round(float(np.mean(puntajes)), 4),
                      "macro_f1_cv_std": round(float(np.std(puntajes)), 4), "n_evaluaciones": len(puntajes)})
    tabla = pd.DataFrame(filas).sort_values("macro_f1_cv_media", ascending=False).reset_index(drop=True)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DIR_MODELOS / "variabilidad_cv_train.csv", index=False)
    return tabla

def analizar_regla_decision(pipeline, X_val, y_val, pesos_dropout=(1.0, 1.25, 1.5, 2.0, 2.5, 3.0)) -> pd.DataFrame:
    """Sensibilidad SOLO en validación: multiplica P(Dropout) por un peso y mide macro-F1 y recalls. No usa la prueba."""
    clases = list(pipeline.named_steps["clasificador"].classes_)
    i_drop = clases.index("Dropout")
    proba = pipeline.predict_proba(X_val)
    filas = []
    for w in pesos_dropout:
        ajustada = proba.copy()
        ajustada[:, i_drop] *= w
        m = calcular_metricas(y_val, np.array(clases)[ajustada.argmax(axis=1)])
        filas.append({"peso_dropout": w, "macro_f1_val": m["macro_f1"],
                      **{f"recall_{c}": m["recall_por_clase"][c] for c in CLASES}})
    tabla = pd.DataFrame(filas)
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(DIR_MODELOS / "regla_decision_validacion.csv", index=False)
    return tabla


def comparar_sin_variables(train: pd.DataFrame, variables_quitar: dict, version: str = "regresion_logistica",
                           n_splits: int = 5) -> pd.DataFrame:
    """CV estratificada SOLO sobre train: macro-F1 con todas las variables y sin cada grupo indicado."""
    pipeline = definir_modelos()[version]
    filas = []
    for nombre, quitar in ({"con_todas": []} | variables_quitar).items():
        variables = [v for v in VARIABLES_MODELO if v not in quitar]
        X, y = separar_xy(train, variables)
        puntajes = []
        for semilla in SEMILLAS_VARIABILIDAD:
            modelo = clone(pipeline)
            if "random_state" in modelo.named_steps["clasificador"].get_params():
                modelo.set_params(clasificador__random_state=semilla)
            from sklearn.model_selection import StratifiedGroupKFold

            columnas_grupo = [
                col for col in VARIABLES_MODELO
                if all(col not in excluir for excluir in variables_quitar.values())
            ]
            grupos = pd.util.hash_pandas_object(train[columnas_grupo], index=False)
            cv = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=semilla
            )
            puntajes.extend(cross_val_score(
                modelo, X, y,
                cv=cv,
                groups=grupos,
                scoring="f1_macro"
            ))
        filas.append({"escenario": nombre, "n_variables": len(variables),
                      "macro_f1_cv_media": round(float(np.mean(puntajes)), 4),
                      "macro_f1_cv_std": round(float(np.std(puntajes)), 4)})
    return pd.DataFrame(filas)
# Metricas validadas por Yuan-S29