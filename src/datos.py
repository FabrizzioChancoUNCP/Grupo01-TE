# src/datos.py
"""Carga, limpieza y partición de datos. No ajusta ningún transformador (sin fuga)."""
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42  # única definición de la semilla; los demás módulos la importan de aquí
RUTA_DATOS = Path(__file__).resolve().parent.parent / "data" / "data.csv"
COLUMNA_OBJETIVO = "Target"
PROP_PRUEBA = 0.15
PROP_VAL = 0.15

CLASES = ["Dropout", "Enrolled", "Graduate"]

# Momento de la predicción: al terminar el PRIMER semestre.
# Fuga: se miden después de ese momento, se excluyen siempre.
COLUMNAS_FUGA = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

# Sensibles: NO entran al modelo; se conservan solo para auditar el error por grupo.
COLUMNAS_SENSIBLES = [
    "Gender",
    "Marital status",
    "Nacionality",  # (sic) así viene escrito en el CSV
    "International",
    "Educational special needs",
]
COLUMNAS_AUDITORIA = ["Gender", "Scholarship holder", "International", "Educational special needs",
                      "Debtor", "Tuition fees up to date"]

# Variables del modelo (8): conocidas al cierre del 1.er semestre y que el chatbot puede preguntar.
VARIABLES_MODELO = [
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Tuition fees up to date",
    "Debtor",
    "Scholarship holder",
    "Age at enrollment",
    "Admission grade",
]

# Campo del contrato de la API -> columna del dataset (un solo lugar).
MAPA_API_A_COLUMNA = {
    "unidades_inscritas_s1": "Curricular units 1st sem (enrolled)",
    "unidades_aprobadas_s1": "Curricular units 1st sem (approved)",
    "promedio_s1": "Curricular units 1st sem (grade)",
    "matricula_al_dia": "Tuition fees up to date",
    "deudor": "Debtor",
    "becario": "Scholarship holder",
    "edad_ingreso": "Age at enrollment",
    "nota_admision": "Admission grade",
}

# Rangos por dominio (no por lo observado): sirven aquí y para validar la API.
RANGOS_DOMINIO = {
    "Curricular units 1st sem (enrolled)": (0, 60),
    "Curricular units 1st sem (approved)": (0, 60),
    "Curricular units 1st sem (grade)": (0, 20),
    "Tuition fees up to date": (0, 1),
    "Debtor": (0, 1),
    "Scholarship holder": (0, 1),
    "Age at enrollment": (16, 80),
    "Admission grade": (0, 200),
}

_ENROLLED_S1 = "Curricular units 1st sem (enrolled)"
_APPROVED_S1 = "Curricular units 1st sem (approved)"
_GRADE_S1 = "Curricular units 1st sem (grade)"


def cargar_datos(ruta: Path = RUTA_DATOS) -> pd.DataFrame:
    """Recibe la ruta del CSV (sep=';') y devuelve el DataFrame crudo."""
    df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    df.columns = df.columns.str.strip()  # decisión: nombres sin espacios ni tabuladores sobrantes
    return df


def eliminar_duplicados(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe un DataFrame y devuelve uno sin filas repetidas (antes de particionar, para no contaminar la prueba)."""
    return df.drop_duplicates().reset_index(drop=True)


def particionar(df: pd.DataFrame, seed: int = SEED):
    """Divide 70/15/15, estratificando y agrupando perfiles idénticos."""
    if COLUMNA_OBJETIVO not in df.columns:
        raise KeyError(f"Falta la columna objetivo '{COLUMNA_OBJETIVO}'")

    columnas = [c for c in VARIABLES_MODELO if c in df.columns]
    if not columnas:
        columnas = [c for c in df.columns if c != COLUMNA_OBJETIVO]

    grupos = pd.util.hash_pandas_object(df[columnas], index=False)
    clases_grupo = df.groupby(grupos)[COLUMNA_OBJETIVO].agg(
        lambda y: y.mode().iloc[0]
    )

    ids_train, ids_reserva = train_test_split(
        clases_grupo.index,
        test_size=PROP_VAL + PROP_PRUEBA,
        stratify=clases_grupo,
        random_state=seed
    )

    ids_val, ids_prueba = train_test_split(
        ids_reserva,
        test_size=PROP_PRUEBA / (PROP_VAL + PROP_PRUEBA),
        stratify=clases_grupo.loc[ids_reserva],
        random_state=seed
    )

    train = df.loc[grupos.isin(ids_train)]
    val = df.loc[grupos.isin(ids_val)]
    prueba = df.loc[grupos.isin(ids_prueba)]

    return train, val, prueba


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe el DataFrame crudo y devuelve una copia con nombres y etiquetas sin espacios sobrantes.

    Lanza ValueError si 'Target' trae clases fuera de CLASES (no se corrigen en silencio).
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    if COLUMNA_OBJETIVO not in df.columns:
        raise KeyError(f"Falta la columna objetivo '{COLUMNA_OBJETIVO}'")
    df[COLUMNA_OBJETIVO] = df[COLUMNA_OBJETIVO].astype(str).str.strip()
    invalidas = sorted(set(df[COLUMNA_OBJETIVO].unique()) - set(CLASES))
    if invalidas:
        raise ValueError(f"Clases desconocidas en '{COLUMNA_OBJETIVO}': {invalidas}")
    return df


def contar_duplicados_conflictivos(df: pd.DataFrame) -> int:
    """Devuelve cuántos perfiles (mismas variables, distinto 'Target') hay: ruido en la etiqueta."""
    atributos = [c for c in df.columns if c != COLUMNA_OBJETIVO]
    repetidas = df[df.duplicated(subset=atributos, keep=False)]
    if repetidas.empty:
        return 0
    etiquetas = repetidas.groupby(atributos, dropna=False)[COLUMNA_OBJETIVO].nunique()
    return int((etiquetas > 1).sum())


def diagnosticar_calidad(df: pd.DataFrame) -> dict:
    """Recibe el DataFrame (columnas ya sin espacios) y devuelve un dict con los problemas de calidad medidos.

    Solo mide, no modifica. Todo valor es un tipo simple (serializable a JSON).
    """
    if COLUMNA_OBJETIVO not in df.columns:
        raise KeyError(f"Falta la columna objetivo '{COLUMNA_OBJETIVO}'")
    fuera = {}
    for col, (minimo, maximo) in RANGOS_DOMINIO.items():
        if col in df.columns:
            fuera[col] = int(((df[col] < minimo) | (df[col] > maximo)).sum())
    nulos = df.isna().sum()
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "nulos_por_columna": {c: int(n) for c, n in nulos.items() if n > 0},
        "duplicados_exactos": int(df.duplicated().sum()),
        "perfiles_con_etiqueta_conflictiva": contar_duplicados_conflictivos(df),
        "clases": {str(k): int(v) for k, v in df[COLUMNA_OBJETIVO].value_counts().items()},
        "proporcion_clases": {str(k): round(float(v), 4) for k, v in df[COLUMNA_OBJETIVO].value_counts(normalize=True).items()},
        "accuracy_si_siempre_clase_mayoritaria": round(float(df[COLUMNA_OBJETIVO].value_counts(normalize=True).max()), 4),
        "fuera_de_rango_dominio": fuera,
        "aprobadas_mayor_que_inscritas_s1": int((df[_APPROVED_S1] > df[_ENROLLED_S1]).sum()),
        "sin_unidades_inscritas_s1": int((df[_ENROLLED_S1] == 0).sum()),
        "promedio_cero_con_unidades_s1": int(((df[_GRADE_S1] == 0) & (df[_ENROLLED_S1] > 0)).sum()),
    }


def separar_xy(df: pd.DataFrame, variables: list | None = None):
    """Recibe el DataFrame y devuelve (X, y) solo con las variables permitidas.

    Falla si se pide una variable de fuga, sensible u objetivo (protege al ampliar la lista).
    """
    variables = list(VARIABLES_MODELO if variables is None else variables)
    prohibidas = set(variables) & (set(COLUMNAS_FUGA) | set(COLUMNAS_SENSIBLES) | {COLUMNA_OBJETIVO})
    if prohibidas:
        raise ValueError(f"Variables no permitidas (fuga, sensibles u objetivo): {sorted(prohibidas)}")
    faltan = [v for v in variables if v not in df.columns]
    if faltan:
        raise KeyError(f"Faltan columnas en los datos: {faltan}")
    return df[variables].copy(), df[COLUMNA_OBJETIVO].copy()


def obtener_grupos(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe el DataFrame y devuelve las columnas de auditoría + 'grupo_edad' (NO son entradas del modelo)."""
    grupos = df[COLUMNAS_AUDITORIA].copy()
    grupos["grupo_edad"] = pd.cut(
        df["Age at enrollment"], bins=[0, 20, 25, 200], labels=["<=20", "21-25", ">25"]
    ).astype(str)
    return grupos


def verificar_sin_fuga(train: pd.DataFrame, val: pd.DataFrame, prueba: pd.DataFrame) -> None:
    """Lanza ValueError si dos particiones comparten índices o filas idénticas (contaminaría la prueba)."""
    partes = {"train": train, "val": val, "prueba": prueba}
    for a, b in [("train", "val"), ("train", "prueba"), ("val", "prueba")]:
        if len(partes[a].index.intersection(partes[b].index)) > 0:
            raise ValueError(f"Fuga: índices compartidos entre {a} y {b}")
        columnas = [c for c in VARIABLES_MODELO if c in partes[a] and c in partes[b]]
        if not columnas:
            columnas = [c for c in partes[a] if c in partes[b] and c != COLUMNA_OBJETIVO]

        comunes = set(pd.util.hash_pandas_object(partes[a][columnas], index=False)) & set(
            pd.util.hash_pandas_object(partes[b][columnas], index=False)
        )
        if comunes:
            raise ValueError(f"Fuga: {len(comunes)} filas con entradas idénticas entre {a} y {b}")


def resumen_particion(train: pd.DataFrame, val: pd.DataFrame, prueba: pd.DataFrame) -> dict:
    """Devuelve tamaños y proporción de clases por partición (para la ficha JSON y el informe)."""
    partes = {"train": train, "val": val, "prueba": prueba}
    return {
        "filas_train": int(len(train)),
        "filas_val": int(len(val)),
        "filas_test": int(len(prueba)),
        "semilla": SEED,
        "proporcion_clases": {
            nombre: {str(k): round(float(v), 4) for k, v in p[COLUMNA_OBJETIVO].value_counts(normalize=True).items()}
            for nombre, p in partes.items()
        },
    }


def cargar_y_particionar(ruta: Path = RUTA_DATOS):
    """Flujo completo: cargar, limpiar, quitar duplicados, particionar y verificar. Devuelve (train, val, prueba)."""
    df = eliminar_duplicados(limpiar(cargar_datos(ruta)))  # duplicados ANTES de particionar
    train, val, prueba = particionar(df)
    verificar_sin_fuga(train, val, prueba)
    return train, val, prueba

def resumen_eda(df: pd.DataFrame) -> dict:
    """Recibe el DataFrame limpio y devuelve las cifras que cita el informe (solo lectura)."""
    res = diagnosticar_calidad(df)
    es_dropout = df[COLUMNA_OBJETIVO] == "Dropout"
    bins = [-1, 0, 2, 4, 6, 100]

    def tasa(grupo):
        return {str(k): round(float(v), 4) for k, v in es_dropout.groupby(grupo, observed=True).mean().items()}

    res["perfiles_conflictivos_8_variables"] = contar_duplicados_conflictivos(df[VARIABLES_MODELO + [COLUMNA_OBJETIVO]])
    res["clases_sin_unidades_s1"] = {str(k): int(v) for k, v in
                                     df.loc[df[_ENROLLED_S1] == 0, COLUMNA_OBJETIVO].value_counts().items()}
    res["tasa_dropout_por_unidades_aprobadas"] = {
        "1er_semestre": tasa(pd.cut(df[_APPROVED_S1], bins)),
        "2do_semestre_FUGA": tasa(pd.cut(df["Curricular units 2nd sem (approved)"], bins)),
    }
    res["tasa_dropout_por_variable"] = {c: tasa(df[c]) for c in ["Tuition fees up to date", "Debtor", "Scholarship holder"]}
    return res

def contar_filas_modelo_compartidas(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Devuelve cuántas filas (solo las 8 variables del modelo) aparecen idénticas en ambas particiones."""
    ha = set(pd.util.hash_pandas_object(a[VARIABLES_MODELO], index=False))
    hb = set(pd.util.hash_pandas_object(b[VARIABLES_MODELO], index=False))
    return len(ha & hb)