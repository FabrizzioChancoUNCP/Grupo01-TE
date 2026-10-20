"""Chatbot de reglas para orientación estudiantil; consulta la API sin entrenar modelos."""
# Flujo conversacional finalizado por QuintoMiguelUNCP

import os
import math
from dataclasses import dataclass, field
from typing import Any

from src.datos import MAPA_API_A_COLUMNA, RANGOS_DOMINIO

import requests


CLASES = ("Dropout", "Enrolled", "Graduate")
PREGUNTAS = (
    ("unidades_inscritas_s1", "¿Cuántas unidades curriculares matriculaste en el primer semestre? (0 a 60)", 0, 60, True),
    ("unidades_aprobadas_s1", "¿Cuántas unidades aprobaste en el primer semestre? (0 a 60)", 0, 60, True),
    ("promedio_s1", "¿Cuál fue tu promedio del primer semestre? (0 a 20)", 0, 20, False),
    ("matricula_al_dia", "¿Tienes tus pagos de matrícula al día? (sí/no)", 0, 1, True),
    ("deudor", "¿Registras alguna deuda con la institución? (sí/no)", 0, 1, True),
    ("becario", "¿Tienes una beca? (sí/no)", 0, 1, True),
    ("edad_ingreso", "¿Qué edad tenías al ingresar? (16 a 80 años)", 16, 80, True),
    ("nota_admision", "¿Cuál fue tu nota de admisión? (0 a 200)", 0, 200, False),
)


# Contrato único de E1-E4: evita replicar claves y rangos inconsistentes.
assert {fila[0] for fila in PREGUNTAS} == set(MAPA_API_A_COLUMNA)
for campo, _, minimo, maximo, _ in PREGUNTAS:
    assert (minimo, maximo) == RANGOS_DOMINIO[MAPA_API_A_COLUMNA[campo]]


class ErrorAPI(Exception):
    """Indica una respuesta inválida de la API o un problema de conexión."""


def interpretar_numero(texto: str, minimo: float, maximo: float, entero: bool) -> int | float:
    """Convierte texto numérico finito a un valor dentro del rango permitido."""
    cadena = texto.strip().replace(",", ".")
    try:
        numero = float(cadena)
    except ValueError as exc:
        raise ValueError("Escribe un número válido.") from exc
    if not math.isfinite(numero) or not minimo <= numero <= maximo:
        raise ValueError(f"El valor debe estar entre {minimo:g} y {maximo:g}.")
    if entero:
        if not numero.is_integer():
            raise ValueError("Escribe un número entero.")
        return int(numero)
    return numero


def interpretar_respuesta(campo: str, texto: str, minimo: float, maximo: float, entero: bool) -> int | float:
    """Valida una respuesta numérica o de sí/no según el campo."""
    if campo in ("matricula_al_dia", "deudor", "becario"):
        normalizado = texto.strip().casefold()
        if normalizado in ("sí", "si", "s", "1"):
            return 1
        if normalizado in ("no", "n", "0"):
            return 0
        raise ValueError("Responde sí o no.")
    return interpretar_numero(texto, minimo, maximo, entero)


def consultar_prediccion(datos: dict[str, int | float], url_base: str, sesion: Any = requests) -> dict:
    """Envía los ocho datos a POST /predecir y devuelve una respuesta validada."""
    try:
        respuesta = sesion.post(f"{url_base.rstrip('/')}/predecir", json=datos, timeout=10)
        if respuesta.status_code == 400:
            try:
                detalle = respuesta.json()
            except ValueError:
                detalle = {}
            mensaje = detalle.get("error", "Datos rechazados por el servicio.") if isinstance(detalle, dict) else "Datos rechazados por el servicio."
            raise ErrorAPI(f"El servicio rechazó los datos: {mensaje}")
        respuesta.raise_for_status()
        resultado = respuesta.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ErrorAPI("El servicio devolvió una respuesta que no es JSON.") from exc
    except requests.HTTPError as exc:
        codigo = exc.response.status_code if exc.response is not None else "desconocido"
        raise ErrorAPI(
            f"El servicio respondió con un error HTTP {codigo}."
        ) from exc
    except requests.RequestException as exc:
        raise ErrorAPI("No se pudo conectar con el servicio de predicción.") from exc
    except ValueError as exc:
        raise ErrorAPI("El servicio devolvió una respuesta que no es JSON.") from exc

    if not isinstance(resultado, dict) or resultado.get("clase") not in CLASES:
        raise ErrorAPI("El servicio devolvió una clase desconocida.")
    probabilidades = resultado.get("probabilidades")
    if not isinstance(probabilidades, dict) or set(probabilidades) != set(CLASES):
        raise ErrorAPI("El servicio no devolvió las tres probabilidades requeridas.")
    for valor in probabilidades.values():
        if isinstance(valor, bool) or not isinstance(valor, (int, float)) or not 0 <= valor <= 1 or not math.isfinite(valor):
            raise ErrorAPI("El servicio devolvió probabilidades inválidas.")
    if abs(sum(probabilidades.values()) - 1) > 0.02:
        raise ErrorAPI("Las probabilidades recibidas no suman aproximadamente 1.")
    if probabilidades[resultado["clase"]] + 0.0001 < max(probabilidades.values()):
        raise ErrorAPI("La clase informada no coincide con las probabilidades.")
    if not isinstance(resultado.get("version"), str) or not resultado["version"].strip():
        raise ErrorAPI("El servicio no informó la versión del modelo.")
    return resultado


def mensaje_resultado(resultado: dict, umbral_alerta: float | None = None) -> str:
    """Comunica la predicción con lenguaje prudente y orientación a tutoría."""
    riesgo = resultado["probabilidades"]["Dropout"]
    categoria = resultado["clase"]
    descripciones = {
        "Dropout": "El sistema identifica señales que merecen atención y acompañamiento.",
        "Enrolled": "El resultado no permite descartar dificultades académicas o necesidad de apoyo.",
        "Graduate": "La categoría estimada es Graduate, pero no garantiza que se complete la carrera.",
    }
    mensaje = (f"Resultado orientativo: {descripciones[categoria]} "
               f"La probabilidad estimada de deserción es {riesgo:.0%}. "
               "Esta estimación no determina tu futuro ni reemplaza la evaluación de una persona.")
    if categoria == "Dropout" or (umbral_alerta is not None and riesgo >= umbral_alerta):
        mensaje += " Te recomendamos conversar pronto con el área de tutoría para recibir apoyo personalizado."
    else:
        mensaje += " Si necesitas orientación, el área de tutoría también puede ayudarte."
        if umbral_alerta is None:
            mensaje += " El umbral de alerta aún no está configurado; se recomienda revisión por tutoría antes de interpretar el riesgo como bajo."
    return mensaje


@dataclass
class Chatbot:
    """Mantiene el estado de la conversación sin guardar identificadores personales."""

    url_base: str = field(default_factory=lambda: os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8000"))
    max_reintentos: int = 3
    umbral_alerta: float | None = None
    sesion: Any = requests
    posicion: int = 0
    reintentos: int = 0
    datos: dict[str, int | float] = field(default_factory=dict)
    terminado: bool = False

    def __post_init__(self) -> None:
        """Valida configuración antes de comenzar."""
        if self.max_reintentos < 1 or (self.umbral_alerta is not None and not 0 <= self.umbral_alerta <= 1):
            raise ValueError("Configuración inválida de reintentos o umbral.")

    def iniciar(self) -> str:
        """Devuelve el saludo y la primera pregunta."""
        if self.terminado:
            return "Esta conversación ha terminado. Inicia una nueva sesión para volver a consultar."
        return ("Hola. Soy un asistente de orientación académica. Haré ocho preguntas "
                "sobre tu primer semestre para ofrecer una estimación orientativa. "
                "No guardo tus respuestas una vez terminada la consulta. No es un diagnóstico ni reemplaza al tutor. "
                "No compartas tu nombre ni datos que te identifiquen. "
                "Puedes escribir 'hablar con un tutor' para recibir orientación humana o 'salir' para cancelar.\n" + PREGUNTAS[self.posicion][1])

    def responder(self, texto: str) -> str:
        """Procesa una respuesta, pregunta el siguiente dato o consulta la API."""
        if self.terminado:
            return "Esta conversación ha terminado. Inicia una nueva sesión para volver a consultar."
        if texto.strip().casefold() in ("hablar con un tutor", "hablar con tutor", "tutor", "tutoría", "tutoria"):
            self.terminado = True
            self.datos.clear()
            return ("Puedes solicitar atención al área de tutoría de tu institución en cualquier momento. "
                    "No he reservado una cita ni enviado tus datos a nadie.")
        if texto.strip().casefold() in ("salir", "cancelar"):
            self.terminado = True
            self.datos.clear()
            return "Consulta cancelada. Si necesitas apoyo, comunícate con el área de tutoría."
        campo, pregunta, minimo, maximo, entero = PREGUNTAS[self.posicion]
        try:
            valor = interpretar_respuesta(campo, texto, minimo, maximo, entero)
            if campo == "unidades_aprobadas_s1" and valor > self.datos["unidades_inscritas_s1"]:
                raise ValueError("Las unidades aprobadas no pueden superar las matriculadas.")
        except ValueError as exc:
            self.reintentos += 1
            if self.reintentos >= self.max_reintentos:
                self.terminado = True
                self.datos.clear()
                return ("No pude interpretar tus respuestas después de varios intentos. "
                        "Por favor, solicita apoyo a una persona del área de tutoría.")
            return f"{exc} Intento {self.reintentos} de {self.max_reintentos}.\n{pregunta}"
        self.datos[campo] = valor
        self.posicion += 1
        self.reintentos = 0
        if self.posicion < len(PREGUNTAS):
            return PREGUNTAS[self.posicion][1]
        self.terminado = True
        try:
            resultado = consultar_prediccion(self.datos.copy(), self.url_base, self.sesion)
            return mensaje_resultado(resultado, self.umbral_alerta)
        except ErrorAPI as exc:
            return (f"No se pudo obtener una predicción en este momento. {exc} "
                    "Puedes consultar al área de tutoría e intentarlo más tarde.")
        finally:
            self.datos.clear()


def ejecutar_chatbot() -> None:
    """Ejecuta el chatbot de reglas desde la terminal."""
    valor_umbral = os.getenv("CHATBOT_UMBRAL_DROPOUT")
    try:
        umbral = float(valor_umbral) if valor_umbral is not None else None
    except ValueError as exc:
        raise ValueError("CHATBOT_UMBRAL_DROPOUT debe ser un número entre 0 y 1.") from exc
    asistente = Chatbot(umbral_alerta=umbral)
    print(asistente.iniciar())
    while not asistente.terminado:
        try:
            respuesta = input("Tú: ")
        except (EOFError, KeyboardInterrupt):
            print("\nConversación cancelada.")
            break
        print("Chatbot:", asistente.responder(respuesta))


if __name__ == "__main__":
    ejecutar_chatbot()
