
from ultralytics import YOLO
from pathlib import Path

# Rutas del modelo y del dataset
MODELO = Path("best.pt")
DATASET_YAML = Path("data.yaml")

def evaluar():
    if not MODELO.exists():
        raise FileNotFoundError(
            "No se encuentra best.pt. "
            "Coloca aquí los pesos del modelo entrenado."
        )

    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            "No se encuentra data.yaml."
        )

    modelo = YOLO(str(MODELO))

    resultados = modelo.val(
        data=str(DATASET_YAML),
        split="test",
        imgsz=512,
        plots=True,
        project="results",
        name="evaluacion_ganaderia"
    )

    print("Evaluación finalizada.")
    print(f"Precisión (mAP50): {resultados.box.map50:.4f}")
    print(f"mAP50-95: {resultados.box.map:.4f}")
    print("Revisa la carpeta results/evaluacion_ganaderia.")

if __name__ == "__main__":
    evaluar()
