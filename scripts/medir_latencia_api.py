"""Mide la latencia HTTP de POST /predecir, sin reentrenar."""
import statistics
import time
import requests

URL = "http://127.0.0.1:8000/predecir"
DATOS = {"unidades_inscritas_s1": 6, "unidades_aprobadas_s1": 5,
         "promedio_s1": 12.5, "matricula_al_dia": 1, "deudor": 0,
         "becario": 1, "edad_ingreso": 19, "nota_admision": 130.0}


def main():
    """Hace 100 solicitudes y muestra mediana y percentil 95 en milisegundos."""
    tiempos = []
    with requests.Session() as sesion:
        for _ in range(100):
            inicio = time.perf_counter()
            respuesta = sesion.post(URL, json=DATOS, timeout=10)
            respuesta.raise_for_status()
            tiempos.append((time.perf_counter() - inicio) * 1000)
    ordenados = sorted(tiempos)
    p95 = ordenados[94]  # percentil 95 por rango más próximo para N=100
    print(f"100 solicitudes; mediana: {statistics.median(tiempos):.2f} ms; p95: {p95:.2f} ms")


if __name__ == "__main__":
    main()
