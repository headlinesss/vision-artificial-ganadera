
from ultralytics import YOLO
from pathlib import Path
import argparse

def detectar(imagen, modelo_path, confianza):
    ruta_imagen = Path(imagen)
    ruta_modelo = Path(modelo_path)

    if not ruta_imagen.exists():
        raise FileNotFoundError(
            f"No se encuentra la imagen: {ruta_imagen}"
        )

    if not ruta_modelo.exists():
        raise FileNotFoundError(
            f"No se encuentra el modelo: {ruta_modelo}"
        )

    modelo = YOLO(str(ruta_modelo))

    resultados = modelo.predict(
        source=str(ruta_imagen),
        conf=confianza,
        imgsz=512,
        save=True,
        project="results",
        name="detecciones"
    )

    for resultado in resultados:
        print("Imagen:", resultado.path)
        for caja in resultado.boxes:
            clase = int(caja.cls[0])
            nombre = modelo.names[clase]
            confianza_objeto = float(caja.conf[0])
            print(
                f"{nombre}: "
                f"{confianza_objeto:.2%} de confianza"
            )

    print("Revisa la carpeta results/detecciones.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detección de bovinos y personas con YOLOv8-seg"
    )
    parser.add_argument("--imagen", required=True)
    parser.add_argument("--modelo", default="best.pt")
    parser.add_argument("--confianza", type=float, default=0.25)
    args = parser.parse_args()

    detectar(args.imagen, args.modelo, args.confianza)
