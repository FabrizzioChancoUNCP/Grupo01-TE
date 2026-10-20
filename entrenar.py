"""Entrenamiento por etapas. La prueba se evalúa UNA sola vez.

  python entrenar.py                -> entrena y promueve usando SOLO validación
  python entrenar.py --final        -> evalúa la prueba UNA vez con la versión promovida
  python entrenar.py --variabilidad -> CV sobre train (no reentrena, no toca fichas ni prueba)
"""
import argparse
from src.datos import cargar_y_particionar, separar_xy
from src.evaluacion import entrenar_y_promover, evaluar_en_prueba, medir_variabilidad
from src.modelo import concluir_con_variabilidad

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true", help="evalúa la prueba una sola vez")
    parser.add_argument("--variabilidad", action="store_true", help="solo CV sobre train")
    args = parser.parse_args()

    train, val, prueba = cargar_y_particionar()

    if args.variabilidad:
        X_train, y_train = separar_xy(train)
        tabla = medir_variabilidad(X_train, y_train)
        print(tabla.to_string(index=False))
        print(concluir_con_variabilidad(tabla))
        return
    
    if args.final:
        print("Métricas en prueba (única evaluación):", evaluar_en_prueba(prueba))
        return

    mejor = entrenar_y_promover(train, val, prueba)   # decide SOLO con validación
    print("Versión promovida:", mejor["version"])


if __name__ == "__main__":
    main()