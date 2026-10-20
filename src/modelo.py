# src/modelo.py
"""Etapa 3: definición de modelos (línea base, modelo simple y MLPs).

Cada modelo es un Pipeline [escalador -> clasificador]. Así el escalado se ajusta (fit)
SOLO con entrenamiento cuando se llama a pipeline.fit(X_train, y_train); validación y
prueba solo se transforman (sin fuga).
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datos import SEED  # semilla única, definida en datos.py


def _con_escalador(clasificador) -> Pipeline:
    """Recibe un clasificador y devuelve un Pipeline con escalado estándar previo."""
    return Pipeline([("escalador", StandardScaler()), ("clasificador", clasificador)])


def definir_modelos(seed: int = SEED) -> dict:
    """Devuelve {identificador_de_version: Pipeline sin entrenar} con las 5 opciones del proyecto."""
    return {
        # Línea base trivial: predice según la distribución de clases de train (no aprende nada).
        "dummy": _con_escalador(DummyClassifier(strategy="prior")),

        # Modelo simple de referencia. class_weight="balanced" compensa el desbalance de 'Enrolled'.
        "regresion_logistica": _con_escalador(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        ),

        # MLP_base: una capa oculta pequeña, sin regularización especial.
        "mlp_base": _con_escalador(
            MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-4,
                          max_iter=500, random_state=seed)
        ),

        # sklearn 1.8.0 falla con early_stopping=True y etiquetas de texto;
        # early_stopping=False mantiene parada por convergencia de pérdida en train.
        # MLP_profunda: dos capas ocultas, parada por estancamiento de pérdida de train.
        "mlp_profunda": _con_escalador(
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                          early_stopping=False, n_iter_no_change=15, max_iter=500,
                          random_state=seed)
        ),

        # MLP_regularizada: L2 fuerte (alpha) + parada por estancamiento de train.
        # Nota: MLPClassifier de sklearn no tiene Dropout; la regularización es L2; se usa parada por estancamiento de pérdida.
        "mlp_regularizada": _con_escalador(
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.1,
                          early_stopping=False, n_iter_no_change=15,
                          max_iter=500, random_state=seed)
        ),
    }


# Hiperparámetros que se documentan en la ficha JSON (los relevantes de cada clasificador).
_PARAMS_FICHA = ["strategy", "C", "class_weight", "max_iter", "hidden_layer_sizes", "activation",
                 "alpha", "early_stopping", "n_iter_no_change", "validation_fraction", "random_state"]


def describir_modelo(pipeline: Pipeline) -> dict:
    """Devuelve la arquitectura y los hiperparámetros del clasificador, listos para guardar en JSON."""
    clf = pipeline.named_steps["clasificador"]
    params = {k: v for k, v in clf.get_params().items() if k in _PARAMS_FICHA}
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = list(params["hidden_layer_sizes"])
    return {"arquitectura": type(clf).__name__, "preprocesador": "StandardScaler", **params}

UMBRAL_DIFERENCIA = 0.01  # diferencias de macro-F1 menores a esto se consideran empate


def concluir_comparacion(fichas: list) -> str:
    """Compara la mejor red con la mejor línea base (solo validación) y devuelve una conclusión honesta."""
    def f1(f):
        return f["metricas_validacion"]["macro_f1"]

    def recall_dropout(f):
        return f["metricas_validacion"]["recall_por_clase"]["Dropout"]

    redes = [f for f in fichas if f["version"].startswith("mlp")]
    bases = [f for f in fichas if not f["version"].startswith("mlp")]
    red, base = max(redes, key=f1), max(bases, key=f1)
    dif = f1(red) - f1(base)

    if dif > UMBRAL_DIFERENCIA:
        veredicto = "la red neuronal SÍ supera a la mejor línea base"
    elif dif < -UMBRAL_DIFERENCIA:
        veredicto = "la red neuronal NO supera a la mejor línea base"
    else:
        veredicto = "la diferencia es menor al umbral: no se puede afirmar que la red sea mejor"

    return (f"Mejor red: {red['version']} (macro-F1 val {f1(red):.4f}, recall Dropout {recall_dropout(red):.4f}). "
            f"Mejor línea base: {base['version']} (macro-F1 val {f1(base):.4f}, recall Dropout {recall_dropout(base):.4f}). "
            f"Diferencia: {dif:+.4f}. En validación, {veredicto}. "
            f"(Umbral de empate: {UMBRAL_DIFERENCIA}; la validación es una sola muestra, no es concluyente por sí sola.)")

def concluir_con_variabilidad(tabla_cv) -> str:
    """Recibe la tabla de medir_variabilidad y compara la diferencia red vs línea base con la desviación (ruido)."""
    cv = tabla_cv.set_index("version")
    redes = [v for v in cv.index if v.startswith("mlp")]
    bases = [v for v in cv.index if not v.startswith("mlp")]
    red = max(redes, key=lambda v: cv.loc[v, "macro_f1_cv_media"])
    base = max(bases, key=lambda v: cv.loc[v, "macro_f1_cv_media"])
    dif = float(cv.loc[red, "macro_f1_cv_media"] - cv.loc[base, "macro_f1_cv_media"])
    ruido = float(max(cv.loc[red, "macro_f1_cv_std"], cv.loc[base, "macro_f1_cv_std"]))
    if abs(dif) <= ruido:
        veredicto = "la diferencia es menor que la desviación: no se puede afirmar que una sea mejor"
    elif dif > 0:
        veredicto = "la red SÍ supera a la línea base por más que el ruido"
    else:
        veredicto = "la red NO supera a la línea base por más que el ruido"
    return (f"CV sobre train: mejor red {red} ({cv.loc[red, 'macro_f1_cv_media']:.4f}±{cv.loc[red, 'macro_f1_cv_std']:.4f}), "
            f"mejor línea base {base} ({cv.loc[base, 'macro_f1_cv_media']:.4f}±{cv.loc[base, 'macro_f1_cv_std']:.4f}). "
            f"Diferencia {dif:+.4f} vs ruido {ruido:.4f}: {veredicto}.")