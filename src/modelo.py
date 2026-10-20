# src/modelo.py
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datos import SEED 


def _con_escalador(clasificador) -> Pipeline:
    return Pipeline([("escalador", StandardScaler()), ("clasificador", clasificador)])


def definir_modelos(seed: int = SEED) -> dict:
    return {
        "dummy": _con_escalador(DummyClassifier(strategy="prior")),

        "regresion_logistica": _con_escalador(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        ),

        "mlp_base": _con_escalador(
            MLPClassifier(hidden_layer_sizes=(32,), activation="relu", alpha=1e-4,
                          max_iter=500, random_state=seed)
        ),

        "mlp_profunda": _con_escalador(
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-4,
                          early_stopping=True, n_iter_no_change=15, max_iter=500,
                          random_state=seed)
        ),

        "mlp_regularizada": _con_escalador(
            MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=0.1,
                          early_stopping=True, n_iter_no_change=15,
                          max_iter=500, random_state=seed)
        ),
    }


_PARAMS_FICHA = ["strategy", "C", "class_weight", "max_iter", "hidden_layer_sizes", "activation",
                 "alpha", "early_stopping", "n_iter_no_change", "validation_fraction", "random_state"]


def describir_modelo(pipeline: Pipeline) -> dict:
    clf = pipeline.named_steps["clasificador"]
    params = {k: v for k, v in clf.get_params().items() if k in _PARAMS_FICHA}
    if "hidden_layer_sizes" in params:
        params["hidden_layer_sizes"] = list(params["hidden_layer_sizes"])
    return {"arquitectura": type(clf).__name__, "preprocesador": "StandardScaler", **params}

UMBRAL_DIFERENCIA = 0.01 


def concluir_comparacion(fichas: list) -> str:
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