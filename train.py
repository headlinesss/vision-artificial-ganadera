
from ultralytics import YOLO
from pathlib import Path

# Configuración del entrenamiento
DATASET_YAML = "data.yaml"
MODELO_BASE = "yolov8n-seg.pt"
EPOCAS = 20
TAMANO_IMAGEN = 512

def entrenar():
    if not Path(DATASET_YAML).exists():
        raise FileNotFoundError(
            "No se encuentra data.yaml. "
            "Coloca el archivo del dataset en la ruta correcta."
        )

    # Cargar el modelo de segmentación YOLOv8 nano
    modelo = YOLO(MODELO_BASE)

    # Entrenar con el dataset etiquetado
    resultados = modelo.train(
        data=DATASET_YAML,
        epochs=EPOCAS,
        imgsz=TAMANO_IMAGEN,
        batch=4,
        project="results",
        name="modelo_ganaderia",
        exist_ok=True
    )

    print("Entrenamiento finalizado.")
    print("Resultados guardados en:", resultados.save_dir)
    print("Revisa weights/best.pt para obtener el mejor modelo.")

if __name__ == "__main__":
    entrenar()
