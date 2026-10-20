"""Prueba extremo a extremo del chatbot contra la API local y el modelo promovido."""
from fastapi.testclient import TestClient
from src.api import app
from src.chatbot import Chatbot


class ClienteLocal:
    """Adaptador de TestClient para POST a la API sin abrir un puerto externo."""

    def __init__(self, cliente):
        self.cliente = cliente

    def post(self, url, json, timeout):
        return self.cliente.post('/predecir', json=json)


def test_chatbot_api_modelo_promovido():
    """Comprueba 8 respuestas y predicción de tres clases con el modelo real."""
    with TestClient(app) as cliente:
        assert cliente.get('/salud').json()['estado'] == 'ok'
        bot = Chatbot(sesion=ClienteLocal(cliente))
        bot.iniciar()
        for respuesta in ('6', '5', '14.5', 'sí', 'no', 'sí', '19', '140'):
            mensaje = bot.responder(respuesta)
        assert bot.terminado
        assert not bot.datos
        assert 'probabilidad estimada de deserción' in mensaje
        assert 'tutoría' in mensaje
