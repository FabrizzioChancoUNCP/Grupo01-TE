"""Pruebas aisladas del chatbot; no necesitan API ni modelo entrenado."""
import pytest
from src.chatbot import Chatbot, ErrorAPI, consultar_prediccion, interpretar_numero

RESPUESTAS = ["6", "5", "14.5", "sí", "no", "sí", "19", "140"]
RESULTADO = {"clase": "Enrolled", "probabilidades": {"Dropout": 0.2, "Enrolled": 0.6, "Graduate": 0.2}, "version": "v2"}


class RespuestaSimulada:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return RESULTADO


class SesionSimulada:
    def __init__(self):
        self.llamadas = []

    def post(self, url, json, timeout):
        self.llamadas.append((url, json, timeout))
        return RespuestaSimulada()


def test_flujo_completo_con_api_simulada():
    sesion = SesionSimulada()
    chatbot = Chatbot(sesion=sesion)
    assert "unidades" in chatbot.iniciar().lower()
    for respuesta in RESPUESTAS:
        mensaje = chatbot.responder(respuesta)
    assert "20%" in mensaje
    assert chatbot.terminado
    assert not chatbot.datos
    assert len(sesion.llamadas) == 1
    assert sesion.llamadas[0][0].endswith("/predecir")
    assert len(sesion.llamadas[0][1]) == 8


def test_iniciar_despues_de_terminar():
    chatbot = Chatbot()
    chatbot.responder("salir")

    assert chatbot.terminado
    assert "terminado" in chatbot.iniciar().lower()


def test_reintentos_y_derivacion():
    chatbot = Chatbot(max_reintentos=2)
    assert "Intento 1" in chatbot.responder("texto no válido")
    assert "tutoría" in chatbot.responder("todavía no entiendo")
    assert chatbot.terminado


def test_error_cruzado_de_unidades():
    chatbot = Chatbot()
    chatbot.responder("4")
    assert "no pueden superar" in chatbot.responder("5")
    assert chatbot.posicion == 1


@pytest.mark.parametrize("valor", ["nan", "inf", "1.5", "61", "hola"])
def test_numero_entero_invalido(valor):
    with pytest.raises(ValueError):
        interpretar_numero(valor, 0, 60, True)


def test_cancelacion():
    chatbot = Chatbot()
    assert "cancelada" in chatbot.responder("salir")
    assert chatbot.terminado


def test_probabilidades_invalidas():
    class Sesion:
        def post(self, *args, **kwargs):
            class Respuesta:
                status_code = 200
                def raise_for_status(self): pass
                def json(self): return {"clase": "Dropout", "probabilidades": {"Dropout": 2, "Enrolled": 0, "Graduate": 0}, "version": "v1"}
            return Respuesta()
    with pytest.raises(ErrorAPI):
        consultar_prediccion({}, "http://localhost:8000", Sesion())


def test_probabilidad_extremadamente_grande(monkeypatch):
    resultado = {
        "clase": "Dropout",
        "probabilidades": {
            "Dropout": 10**400,
            "Enrolled": 0,
            "Graduate": 0
        },
        "version": "v1"
    }
    monkeypatch.setattr(RespuestaSimulada, "json", lambda self: resultado)

    with pytest.raises(ErrorAPI, match="probabilidades inválidas"):
        consultar_prediccion({}, "http://localhost:8000", SesionSimulada())


def test_error_http_503(monkeypatch):
    import requests

    def responder_503(self):
        respuesta = requests.Response()
        respuesta.status_code = 503
        raise requests.HTTPError(response=respuesta)

    monkeypatch.setattr(RespuestaSimulada, "raise_for_status", responder_503)

    with pytest.raises(ErrorAPI, match="HTTP 503"):
        consultar_prediccion({}, "http://localhost:8000", SesionSimulada())


def test_respuesta_json_invalida(monkeypatch):
    import requests

    def json_invalido(self):
        respuesta = requests.Response()
        respuesta.status_code = 200
        respuesta._content = b"respuesta no JSON"
        return respuesta.json()

    monkeypatch.setattr(RespuestaSimulada, "json", json_invalido)

    with pytest.raises(ErrorAPI, match="respuesta que no es JSON"):
        consultar_prediccion({}, "http://localhost:8000", SesionSimulada())


def test_derivacion_por_alto_riesgo():
    from src.chatbot import mensaje_resultado
    resultado = {"clase": "Enrolled", "probabilidades": {"Dropout": 0.7, "Enrolled": 0.2, "Graduate": 0.1}, "version": "v1"}
    assert "conversar pronto" in mensaje_resultado(resultado, umbral_alerta=0.6)


def test_variables_coinciden_con_datos():
    from src.chatbot import PREGUNTAS
    from src.datos import MAPA_API_A_COLUMNA, RANGOS_DOMINIO
    assert {p[0] for p in PREGUNTAS} == set(MAPA_API_A_COLUMNA)
    assert all((p[2], p[3]) == RANGOS_DOMINIO[MAPA_API_A_COLUMNA[p[0]]] for p in PREGUNTAS)


def test_umbral_no_asumido():
    from src.chatbot import mensaje_resultado
    respuesta = {"clase":"Enrolled", "probabilidades":{"Dropout":0.6,"Enrolled":0.3,"Graduate":0.1}, "version":"v1"}
    assert "aún no está configurado" in mensaje_resultado(respuesta)
    assert "conversar pronto" in mensaje_resultado(respuesta, umbral_alerta=0.55)


def test_no_solicita_identificadores():
    mensaje = Chatbot().iniciar().lower()
    assert "no compartas tu nombre" in mensaje
    assert "diagnóstico" in mensaje


def test_derivacion_voluntaria_en_cualquier_paso():
    chatbot = Chatbot()
    chatbot.responder("6")
    mensaje = chatbot.responder("hablar con un tutor")
    assert "tutoría" in mensaje
    assert chatbot.terminado and chatbot.datos == {}
    assert "hablar con un tutor" in Chatbot().iniciar()
