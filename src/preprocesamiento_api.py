# Preprocesamiento validado por Yuan-S29
import math

import pandas as pd

from src.datos import MAPA_API_A_COLUMNA, RANGOS_DOMINIO, VARIABLES_MODELO

CAMPOS_ENTEROS = {"unidades_inscritas_s1", "unidades_aprobadas_s1", "matricula_al_dia",
                  "deudor", "becario", "edad_ingreso"}
CAMPOS_BINARIOS = {"matricula_al_dia", "deudor", "becario"}


class ErrorValidacion(ValueError):
    """Error de entrada con mensaje comprensible y el campo que lo causó."""

    def __init__(self, mensaje: str, campo: str):
        super().__init__(mensaje)
        self.mensaje = mensaje
        self.campo = campo


def validar_entrada(datos) -> dict:
    """Recibe el JSON ya parseado y devuelve {campo_api: valor} limpio; lanza ErrorValidacion si algo falla."""
    if not isinstance(datos, dict):
        raise ErrorValidacion("El cuerpo debe ser un objeto JSON con los 8 campos.", "cuerpo")

    campos_no_permitidos = set(datos) - set(MAPA_API_A_COLUMNA)
    if campos_no_permitidos:
        campo = sorted(campos_no_permitidos)[0]
        raise ErrorValidacion(
            f"El campo '{campo}' no está permitido.",
            campo
        )

    limpios = {}
    for campo, columna in MAPA_API_A_COLUMNA.items():
        if datos.get(campo) is None:
            raise ErrorValidacion(f"Falta el campo obligatorio '{campo}'.", campo)
        valor = datos[campo]
        if isinstance(valor, bool) or not isinstance(valor, (int, float)):
            raise ErrorValidacion(f"'{campo}' debe ser un número.", campo)
        if isinstance(valor, float) and not math.isfinite(valor):
            raise ErrorValidacion(f"'{campo}' debe ser un número finito.", campo)
        if campo in CAMPOS_ENTEROS:
            if isinstance(valor, float) and not valor.is_integer():
                raise ErrorValidacion(f"'{campo}' debe ser un número entero.", campo)
            valor = int(valor)
        else:
            try:
                valor = float(valor)
            except OverflowError as exc:
                raise ErrorValidacion(
                    f"'{campo}' supera el rango numérico permitido.",
                    campo
                ) from exc

        minimo, maximo = RANGOS_DOMINIO[columna]
        if not minimo <= valor <= maximo:
            if campo in CAMPOS_BINARIOS:
                raise ErrorValidacion(f"'{campo}' debe ser 0 o 1.", campo)
            raise ErrorValidacion(f"'{campo}' debe estar entre {minimo} y {maximo}.", campo)
        limpios[campo] = valor

    # Coherencia (en los datos reales no hay ningún caso con aprobadas > inscritas).
    if limpios["unidades_aprobadas_s1"] > limpios["unidades_inscritas_s1"]:
        raise ErrorValidacion("'unidades_aprobadas_s1' no puede superar a 'unidades_inscritas_s1'.",
                              "unidades_aprobadas_s1")
    return limpios


def a_dataframe(valores: dict) -> pd.DataFrame:
    """Recibe {campo_api: valor} validado y devuelve un DataFrame de 1 fila con las columnas del modelo, en orden."""
    fila = {MAPA_API_A_COLUMNA[c]: v for c, v in valores.items()}
    return pd.DataFrame([fila], columns=VARIABLES_MODELO)


def validar_y_preparar(datos) -> pd.DataFrame:
    """Valida el JSON de entrada y devuelve el DataFrame listo para el modelo."""
    return a_dataframe(validar_entrada(datos))
# Preprocesamiento validado por Yuan-S29