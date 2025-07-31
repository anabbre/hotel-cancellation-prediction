import subprocess
import sys

def main():
    """
    Ejecuta en orden los módulos principales del pipeline.

    Llama a train.py, tune.py y evaluate_final.py para realizar
    todo el flujo: entrenamiento, ajuste y evaluación final de modelos.
    """
    modules = ["src.train", "src.tune", "src.evaluate_final"]
    for mod in modules:
        print(f"\n=== Ejecutando: python -m {mod} ===")
        subprocess.run([sys.executable, "-m", mod], check=True)

if __name__ == "__main__":
    main()
