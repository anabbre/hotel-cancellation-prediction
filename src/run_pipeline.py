import subprocess
import sys

def main():
    modules = ["src.train", "src.tune", "src.evaluate_final"]
    for mod in modules:
        print(f"\n=== Ejecutando: python -m {mod} ===")
        subprocess.run([sys.executable, "-m", mod], check=True)

if __name__ == "__main__":
    main()
