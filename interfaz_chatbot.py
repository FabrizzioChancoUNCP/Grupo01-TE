"""Interfaz gráfica (Tkinter) para el chatbot de orientación académica.

No reemplaza la lógica ni las validaciones: muestra el flujo de ``Chatbot``
en una ventana y procesa la consulta HTTP sin bloquear la interfaz.
Iniciar con ``python interfaz_chatbot.py`` desde la carpeta principal del proyecto.
"""

from __future__ import annotations

import math
import os
import queue
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Callable

from src.chatbot import Chatbot, PREGUNTAS


FONDO = "#F5F6FC"
BLANCO = "#FFFFFF"
TEXTO = "#1D2545"
TEXTO_SUAVE = "#737F9D"
AZUL = "#3865F6"
MORADO = "#873AFF"
MENSAJE_BOT = "#F0F2FA"
BORDE = "#E4E9F5"
FUENTE = "DejaVu Sans"


def _mezclar(inicio: str, fin: str, proporcion: float) -> str:
    """Interpola dos colores hexadecimales para dibujar degradados."""
    a = tuple(int(inicio[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(fin[i:i + 2], 16) for i in (1, 3, 5))
    rgb = (round(x + (y - x) * proporcion) for x, y in zip(a, b))
    return "#" + "".join(f"{componente:02x}" for componente in rgb)


def _redondear(canvas: tk.Canvas, x1: float, y1: float, x2: float, y2: float,
               radio: int, relleno: str, etiqueta: str = "fondo") -> None:
    """Rectángulo con esquinas redondeadas, sin dependencias de terceros."""
    radio = max(0, min(radio, int((x2 - x1) / 2), int((y2 - y1) / 2)))
    puntos = [
        x1 + radio, y1, x2 - radio, y1,
        x2, y1, x2, y1 + radio,
        x2, y2 - radio, x2, y2,
        x2 - radio, y2, x1 + radio, y2,
        x1, y2, x1, y2 - radio,
        x1, y1 + radio, x1, y1,
    ]
    canvas.create_polygon(puntos, smooth=True, splinesteps=18,
                          fill=relleno, outline="", tags=etiqueta)


def obtener_umbral_entorno() -> float | None:
    """Respeta la variable de entorno que utiliza el chatbot de terminal."""
    cadena = os.getenv("CHATBOT_UMBRAL_DROPOUT")
    if cadena is None or not cadena.strip():
        return None
    try:
        umbral = float(cadena)
    except ValueError as exc:
        raise ValueError("CHATBOT_UMBRAL_DROPOUT debe estar entre 0 y 1.") from exc
    if not math.isfinite(umbral) or not 0 <= umbral <= 1:
        raise ValueError("CHATBOT_UMBRAL_DROPOUT debe estar entre 0 y 1.")
    return umbral


class InterfazChatbot(tk.Tk):
    """Ventana de conversación; los datos permanecen en el Chatbot original."""

    def __init__(self, fabrica_chatbot: Callable[[], Chatbot] | None = None) -> None:
        super().__init__()
        self.title("Orienta · Asistente virtual académico")
        self.geometry("1190x790")
        self.minsize(680, 620)
        self.configure(bg=FONDO)
        self._fabrica_chatbot = fabrica_chatbot or (lambda: Chatbot(umbral_alerta=obtener_umbral_entorno()))
        self.bot = self._fabrica_chatbot()
        self._sesion = 0
        self._ocupado = False
        self._resultados: queue.Queue[tuple[int, str]] = queue.Queue()
        self._mostrar_laterales = True
        self._ventana_abierta = True

        self._construir_ventana()
        self._nueva_conversacion(inicial=True)
        self.after(80, self._revisar_respuestas)
        self.bind("<Configure>", self._ajustar_laterales)
        self.protocol("WM_DELETE_WINDOW", self._cerrar)

    def _construir_ventana(self) -> None:
        marco = tk.Frame(self, bg=FONDO)
        marco.pack(fill="both", expand=True, padx=22, pady=(14, 10))

        cabecera = tk.Frame(marco, bg=FONDO)
        cabecera.pack(fill="x", pady=(0, 13))
        logo = tk.Canvas(cabecera, width=44, height=44, bg=FONDO, highlightthickness=0)
        logo.pack(side="left", padx=(0, 11))
        _redondear(logo, 2, 2, 42, 42, 12, AZUL)
        logo.create_text(22, 22, text="✦", font=(FUENTE, 24, "bold"), fill=BLANCO)
        titulos = tk.Frame(cabecera, bg=FONDO)
        titulos.pack(side="left")
        tk.Label(titulos, text="ORIENTA", font=(FUENTE, 15, "bold"), fg=TEXTO, bg=FONDO).pack(anchor="w")
        tk.Label(titulos, text="Tu acompañante académico digital", font=(FUENTE, 9),
                 fg=TEXTO_SUAVE, bg=FONDO).pack(anchor="w")
        tk.Label(cabecera, text="ORIENTACIÓN ESTUDIANTIL", font=(FUENTE, 9, "bold"),
                 fg=MORADO, bg=FONDO).pack(side="right", padx=5)

        self.escenario = tk.Frame(marco, bg=FONDO)
        self.escenario.pack(fill="both", expand=True)
        self.lateral_izquierdo = tk.Canvas(self.escenario, width=192, bg=FONDO, highlightthickness=0)
        self.lateral_izquierdo.pack(side="left", fill="y", padx=(0, 20))
        self.lateral_izquierdo.bind("<Configure>", lambda e: self._dibujar_badges(e, izquierda=True))

        self.tarjeta = tk.Frame(self.escenario, bg=BLANCO, highlightthickness=1,
                               highlightbackground=BORDE)
        self.tarjeta.pack(side="left", fill="both", expand=True)
        self.lateral_derecho = tk.Canvas(self.escenario, width=192, bg=FONDO, highlightthickness=0)
        self.lateral_derecho.pack(side="left", fill="y", padx=(20, 0))
        self.lateral_derecho.bind("<Configure>", lambda e: self._dibujar_badges(e, izquierda=False))

        self.encabezado_chat = tk.Canvas(self.tarjeta, height=104, bg=AZUL,
                                        highlightthickness=0)
        self.encabezado_chat.pack(fill="x")
        self.encabezado_chat.bind("<Configure>", self._dibujar_cabecera_chat)

        progreso = tk.Frame(self.tarjeta, bg=BLANCO)
        progreso.pack(fill="x", padx=24, pady=(15, 8))
        superior = tk.Frame(progreso, bg=BLANCO)
        superior.pack(fill="x", pady=(0, 8))
        self.etiqueta_progreso = tk.Label(superior, text="Pregunta 1 de 8", font=(FUENTE, 10, "bold"),
                                         bg=BLANCO, fg=TEXTO)
        self.etiqueta_progreso.pack(side="left")
        self.etiqueta_porcentaje = tk.Label(superior, text="0% completado", font=(FUENTE, 9),
                                           bg=BLANCO, fg=TEXTO_SUAVE)
        self.etiqueta_porcentaje.pack(side="right")
        self.barra_progreso = tk.Canvas(progreso, height=7, bg=BLANCO, highlightthickness=0)
        self.barra_progreso.pack(fill="x")
        self.barra_progreso.bind("<Configure>", lambda e: self._actualizar_progreso())

        # Canvas desplazable con burbujas. No usa la entrada/salida de la terminal.
        conversacion = tk.Frame(self.tarjeta, bg=BLANCO)
        conversacion.pack(fill="both", expand=True, padx=(13, 8), pady=(4, 5))
        self.desplazamiento = tk.Scrollbar(conversacion, orient="vertical", width=9)
        self.desplazamiento.pack(side="right", fill="y")
        self.lienzo_chat = tk.Canvas(conversacion, bg=BLANCO, highlightthickness=0,
                                     yscrollcommand=self.desplazamiento.set)
        self.lienzo_chat.pack(side="left", fill="both", expand=True)
        self.desplazamiento.config(command=self.lienzo_chat.yview)
        self.mensajes = tk.Frame(self.lienzo_chat, bg=BLANCO)
        self._ventana_mensajes = self.lienzo_chat.create_window((0, 0), window=self.mensajes,
                                                                 anchor="nw")
        self.mensajes.bind("<Configure>", self._actualizar_desplazamiento)
        self.lienzo_chat.bind("<Configure>", self._redimensionar_mensajes)
        self.lienzo_chat.bind("<MouseWheel>", self._rueda)
        self.mensajes.bind("<MouseWheel>", self._rueda)

        # Opciones rápidas para las tres preguntas de sí/no.
        self.respuestas_rapidas = tk.Frame(self.tarjeta, bg=BLANCO)
        self.respuestas_rapidas.pack(fill="x", padx=24, pady=(2, 4))
        self.etiqueta_rapida = tk.Label(self.respuestas_rapidas, text="RESPUESTA RÁPIDA",
                                        bg=BLANCO, fg=TEXTO_SUAVE, font=(FUENTE, 8, "bold"))
        self.boton_si = self._boton(self.respuestas_rapidas, "✓  Sí", "#EDF0FF", AZUL,
                                    lambda: self.enviar("sí"))
        self.boton_no = self._boton(self.respuestas_rapidas, "✕  No", "#F2F3F9", TEXTO,
                                    lambda: self.enviar("no"))

        separador = tk.Frame(self.tarjeta, bg=BORDE, height=1)
        separador.pack(fill="x", padx=20, pady=(7, 0))
        pie = tk.Frame(self.tarjeta, bg=BLANCO)
        pie.pack(fill="x", padx=18, pady=(13, 15))
        fila_entrada = tk.Frame(pie, bg="#F1F3FA", padx=12, pady=9)
        fila_entrada.pack(fill="x")
        self.campo_respuesta = tk.Entry(fila_entrada, font=(FUENTE, 11), fg=TEXTO,
                                        bg="#F1F3FA", relief="flat", bd=0,
                                        insertbackground=AZUL)
        self.campo_respuesta.pack(side="left", fill="x", expand=True, ipady=7)
        self.campo_respuesta.bind("<Return>", self._enviar_con_enter)
        self.boton_enviar = self._boton(fila_entrada, "➤", AZUL, BLANCO,
                                        self.enviar, fuente=(FUENTE, 18, "bold"), ancho=4)
        self.boton_enviar.pack(side="right", padx=(11, 0), ipady=4)

        acciones = tk.Frame(pie, bg=BLANCO)
        acciones.pack(fill="x", pady=(11, 0))
        self.boton_tutor = self._boton(acciones, "♡  Hablar con tutor", BLANCO,
                                       MORADO, lambda: self.enviar("hablar con un tutor"),
                                       fuente=(FUENTE, 9, "bold"))
        self.boton_tutor.pack(side="left", padx=(0, 9), ipady=5)
        self.boton_cancelar = self._boton(acciones, "Cancelar", BLANCO, TEXTO_SUAVE,
                                         lambda: self.enviar("salir"), fuente=(FUENTE, 9))
        self.boton_cancelar.pack(side="left", ipady=5)
        self.boton_nuevo = self._boton(acciones, "↻  Nuevo chat", "#F1EDFF", MORADO,
                                      self._nueva_conversacion, fuente=(FUENTE, 9, "bold"))
        self.boton_nuevo.pack(side="right", ipady=5)

        self.estado = tk.Label(marco, text="Tus respuestas se utilizan únicamente durante esta consulta.",
                               font=(FUENTE, 9), fg=TEXTO_SUAVE, bg=FONDO)
        self.estado.pack(pady=(9, 0))

    @staticmethod
    def _boton(padre: tk.Widget, texto: str, fondo: str, tinta: str, accion: Callable,
               fuente: tuple | None = None, ancho: int | None = None) -> tk.Button:
        kwargs = {"width": ancho} if ancho else {}
        return tk.Button(padre, text=texto, command=accion, bg=fondo, fg=tinta,
                         activebackground=fondo, activeforeground=tinta,
                         relief="flat", bd=0, cursor="hand2", padx=10, pady=4,
                         font=fuente or (FUENTE, 10, "bold"), **kwargs)

    def _dibujar_cabecera_chat(self, evento: tk.Event) -> None:
        lienzo = self.encabezado_chat
        ancho = evento.width
        alto = evento.height
        lienzo.delete("all")
        pasos = 105
        for i in range(pasos):
            inicio = round(i * ancho / pasos)
            fin = round((i + 1) * ancho / pasos) + 1
            lienzo.create_rectangle(inicio, 0, fin, alto, fill=_mezclar("#2B73FF", "#9236F8", i / pasos), outline="")
        lienzo.create_oval(22, 24, 74, 76, fill="#EAF0FF", outline="")
        lienzo.create_text(48, 50, text="✦", font=(FUENTE, 27, "bold"), fill=AZUL)
        lienzo.create_text(90, 33, text="Asistente virtual", font=(FUENTE, 18, "bold"),
                            anchor="w", fill=BLANCO)
        lienzo.create_text(91, 66, text="Siempre aquí para orientarte", font=(FUENTE, 10),
                            anchor="w", fill="#F0EEFF")
        if ancho >= 455:
            _redondear(lienzo, ancho - 116, 33, ancho - 16, 65, 16, "#FFFFFF", "activo")
            lienzo.create_oval(ancho - 105, 46, ancho - 96, 55, fill="#27BF88", outline="")
            lienzo.create_text(ancho - 89, 49, text="ACTIVO", fill="#495473",
                                anchor="w", font=(FUENTE, 8, "bold"))

    def _dibujar_badges(self, evento: tk.Event, izquierda: bool) -> None:
        lienzo: tk.Canvas = evento.widget
        ancho, alto = evento.width, evento.height
        lienzo.delete("all")
        items = (("◷", "DISPONIBLE", "24/7"), ("♡", "APOYO", "ACADÉMICO")) if izquierda else (
            ("⚡", "RESPUESTAS", "RÁPIDAS"), ("✓", "ORIENTACIÓN", "SEGURA"))
        for ind, (icono, linea1, linea2) in enumerate(items):
            x, y, radio = ancho / 2, alto * (0.28 if ind == 0 else 0.72), 79
            # Degradado recortado por las ecuaciones del círculo.
            for j in range(-radio, radio + 1, 2):
                mitad = math.sqrt(max(0, radio * radio - j * j))
                color = _mezclar("#B22EF9", "#6736FB", (j + radio) / (radio * 2))
                lienzo.create_line(x - mitad, y + j, x + mitad, y + j,
                                   fill=color, width=3)
            lienzo.create_text(x, y - 31, text=icono, font=(FUENTE, 36, "bold"), fill=BLANCO)
            lienzo.create_text(x, y + 17, text=linea1, font=(FUENTE, 10, "bold"), fill=BLANCO)
            lienzo.create_text(x, y + 38, text=linea2, font=(FUENTE, 12, "bold"), fill=BLANCO)

    def _ajustar_laterales(self, evento: tk.Event) -> None:
        if evento.widget is not self:
            return
        hay_espacio = evento.width >= 1050
        if hay_espacio == self._mostrar_laterales:
            return
        self._mostrar_laterales = hay_espacio
        if hay_espacio:
            self.lateral_izquierdo.pack(side="left", fill="y", padx=(0, 20), before=self.tarjeta)
            self.lateral_derecho.pack(side="left", fill="y", padx=(20, 0))
        else:
            self.lateral_izquierdo.pack_forget()
            self.lateral_derecho.pack_forget()

    def _actualizar_desplazamiento(self, _evento: tk.Event | None = None) -> None:
        self.lienzo_chat.configure(scrollregion=self.lienzo_chat.bbox("all"))
        self.after_idle(lambda: self.lienzo_chat.yview_moveto(1.0) if self._ventana_abierta else None)

    def _redimensionar_mensajes(self, evento: tk.Event) -> None:
        self.lienzo_chat.itemconfigure(self._ventana_mensajes, width=evento.width)

    def _rueda(self, evento: tk.Event) -> None:
        if evento.delta:
            self.lienzo_chat.yview_scroll(-1 if evento.delta > 0 else 1, "units")

    def _mensaje(self, texto: str, usuario: bool = False) -> None:
        fila = tk.Frame(self.mensajes, bg=BLANCO)
        fila.pack(fill="x", padx=9, pady=7)
        if not usuario:
            icono = tk.Label(fila, text="✦", font=(FUENTE, 13, "bold"), fg=AZUL,
                             bg="#EAF0FF", width=3, height=2)
            icono.pack(side="left", anchor="n", padx=(0, 10))

        burbuja = tk.Canvas(fila, bg=BLANCO, highlightthickness=0)
        # Un globo de texto real cuyo alto se adapta automáticamente al contenido.
        color = AZUL if usuario else MENSAJE_BOT
        tinta = BLANCO if usuario else TEXTO
        identificador = burbuja.create_text(18, 15, text=texto, anchor="nw", width=350,
                                             font=(FUENTE, 10), fill=tinta)
        bounds = burbuja.bbox(identificador)
        ancho_texto = (bounds[2] - bounds[0]) if bounds else 120
        alto_texto = (bounds[3] - bounds[1]) if bounds else 20
        ancho_burbuja = min(391, max(110, ancho_texto + 37))
        alto_burbuja = max(50, alto_texto + 31)
        burbuja.configure(width=ancho_burbuja, height=alto_burbuja)
        _redondear(burbuja, 1, 1, ancho_burbuja - 2, alto_burbuja - 2, 17, color)
        burbuja.tag_lower("fondo", identificador)
        burbuja.pack(side="right" if usuario else "left", anchor="n")
        self._actualizar_desplazamiento()

    def _actualizar_progreso(self) -> None:
        total = len(PREGUNTAS)
        contestadas = min(self.bot.posicion, total)
        completado = self.bot.terminado and contestadas == total
        if completado:
            texto = "Consulta completada"
        elif self.bot.terminado:
            texto = "Consulta finalizada"
        else:
            texto = f"Pregunta {contestadas + 1} de {total}"
        self.etiqueta_progreso.configure(text=texto)
        self.etiqueta_porcentaje.configure(text=f"{round(contestadas / total * 100)}% completado")
        lienzo = self.barra_progreso
        lienzo.delete("all")
        ancho = lienzo.winfo_width()
        lienzo.create_rectangle(0, 0, ancho, 7, fill="#EDECF8", outline="")
        if contestadas:
            lienzo.create_rectangle(0, 0, ancho * contestadas / total, 7, fill=MORADO, outline="")

    def _actualizar_controles(self) -> None:
        disponible = not self._ocupado and not self.bot.terminado
        self.campo_respuesta.configure(state="normal" if disponible else "disabled")
        self.boton_enviar.configure(state="normal" if disponible else "disabled")
        self.boton_tutor.configure(state="normal" if disponible else "disabled")
        self.boton_cancelar.configure(state="normal" if disponible else "disabled")
        self.boton_si.pack_forget()
        self.boton_no.pack_forget()
        self.etiqueta_rapida.pack_forget()
        if disponible and PREGUNTAS[self.bot.posicion][0] in {"matricula_al_dia", "deudor", "becario"}:
            self.etiqueta_rapida.pack(side="left", padx=(0, 13))
            self.boton_si.pack(side="left", padx=(0, 7))
            self.boton_no.pack(side="left")
        if disponible:
            self.campo_respuesta.focus_set()
        self._actualizar_progreso()

    def _nueva_conversacion(self, inicial: bool = False) -> None:
        if not inicial:
            self._sesion += 1
            self.bot = self._fabrica_chatbot()
        self._ocupado = False
        self.campo_respuesta.configure(state="normal")
        self.campo_respuesta.delete(0, "end")
        for widget in self.mensajes.winfo_children():
            widget.destroy()
        self.lienzo_chat.yview_moveto(0)
        saludo = self.bot.iniciar()
        introduccion, pregunta = saludo.rsplit("\n", 1)
        self._mensaje(introduccion)
        self._mensaje(pregunta)
        self.estado.configure(text="Responde las 8 preguntas sin proporcionar tu nombre ni otros identificadores.")
        self._actualizar_controles()

    def _enviar_con_enter(self, _evento: tk.Event) -> str:
        self.enviar()
        return "break"

    def enviar(self, respuesta: str | None = None) -> None:
        """Envía una respuesta sin congelar la ventana al realizar POST /predecir."""
        if self._ocupado or self.bot.terminado:
            return
        texto = self.campo_respuesta.get().strip() if respuesta is None else respuesta.strip()
        if not texto:
            return
        self.campo_respuesta.delete(0, "end")
        self._mensaje(texto, usuario=True)
        self._ocupado = True
        ultima = self.bot.posicion == len(PREGUNTAS) - 1 and texto.casefold() not in {
            "salir", "cancelar", "tutor", "tutoría", "tutoria", "hablar con un tutor", "hablar con tutor"}
        self.estado.configure(text="Consultando la API de predicción..." if ultima
                              else "Procesando tu respuesta...")
        self._actualizar_controles()
        sesion = self._sesion
        bot = self.bot
        threading.Thread(target=self._procesar_en_hilo, args=(sesion, bot, texto), daemon=True).start()

    def _procesar_en_hilo(self, sesion: int, bot: Chatbot, texto: str) -> None:
        try:
            respuesta = bot.responder(texto)
        except Exception:
            # No mostrar detalles internos ni rastros de excepciones en pantalla.
            respuesta = "Ocurrió un problema inesperado. Puedes iniciar un nuevo chat o hablar con tutoría."
        self._resultados.put((sesion, respuesta))

    def _revisar_respuestas(self) -> None:
        if not self._ventana_abierta:
            return
        try:
            while True:
                sesion, respuesta = self._resultados.get_nowait()
                if sesion != self._sesion:
                    continue  # Ignorar respuestas de una conversación reiniciada.
                self._ocupado = False
                self._mensaje(respuesta)
                if self.bot.terminado:
                    self.estado.configure(text="Conversación finalizada. Puedes seleccionar «Nuevo chat».")
                else:
                    self.estado.configure(text="Tus respuestas no se almacenan después de la consulta.")
                self._actualizar_controles()
        except queue.Empty:
            pass
        self.after(80, self._revisar_respuestas)

    def _cerrar(self) -> None:
        self._ventana_abierta = False
        self.destroy()


def main() -> None:
    """Abre la aplicación. La API puede iniciarse con el lanzador principal."""
    try:
        app = InterfazChatbot()
    except (ValueError, tk.TclError) as exc:
        # Mensaje legible tanto si falta servidor gráfico como si la configuración es inválida.
        raise SystemExit(f"No se pudo abrir la interfaz del chatbot: {exc}") from exc
    app.mainloop()



# Este archivo puede colocarse directamente en la raíz del proyecto.
from urllib.parse import urlparse
import requests


def iniciar_api_local_si_necesaria() -> None:
    url = os.getenv("CHATBOT_API_URL", "http://127.0.0.1:8000")
    direccion = urlparse(url)
    if direccion.scheme != "http" or direccion.hostname not in ("127.0.0.1", "localhost"):
        return  # Se respeta cualquier URL remota configurada.
    try:
        response = requests.get(f"{url.rstrip('/')}/salud", timeout=0.5)
        if response.ok:
            return  # El usuario ya tiene una API disponible.
    except requests.RequestException:
        pass

    import uvicorn

    def ejecutar_api() -> None:
        config = uvicorn.Config("src.api:app", host=direccion.hostname,
                                port=direccion.port or 80, log_level="error")
        servidor = uvicorn.Server(config)
        servidor.run()

    threading.Thread(target=ejecutar_api, name="api-orienta", daemon=True).start()



if __name__ == "__main__":
    iniciar_api_local_si_necesaria()
    main()
