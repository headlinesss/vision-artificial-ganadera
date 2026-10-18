
# Sistema de Visión Artificial Ganadera

## 1. Descripción del proyecto

Este proyecto implementa un sistema de visión artificial para la detección y segmentación de bovinos y personas mediante YOLOv8n-seg.

El sistema busca identificar los objetos presentes en una imagen, delimitar sus regiones mediante máscaras de segmentación y mostrar la confianza de cada predicción.

## 2. Objetivo

Desarrollar y evaluar un modelo de visión artificial capaz de reconocer bovinos y personas en imágenes, como base para futuras aplicaciones de vigilancia ganadera.

## 3. Tecnologías utilizadas

- Python
- Ultralytics YOLOv8n-seg
- PyTorch
- Google Colab
- Roboflow
- OpenCV
- Matplotlib

## 4. Clases detectadas

| ID | Clase |
|---|---|
| 0 | cow |
| 1 | person |

## 5. Estructura del proyecto

- `train.py`: entrenamiento del modelo.
- `val.py`: evaluación del modelo.
- `detect.py`: inferencia sobre imágenes nuevas.
- `requisitos.txt`: dependencias principales de Python.
- `data.yaml`: configuración del conjunto de datos.
- `best.pt`: pesos del mejor modelo entrenado.

Los archivos del dataset y los pesos del modelo deben colocarse en las rutas correspondientes antes de ejecutar los scripts.

## 6. Instalación

Instalar Python y las dependencias indicadas en `requisitos.txt`:

```bash
pip install -r requisitos.txt
```

## 7. Entrenamiento

Con el dataset correctamente organizado y `data.yaml` configurado:

```bash
python train.py
```

## 8. Evaluación

Con el archivo `best.pt` disponible:

```bash
python val.py
```

## 9. Detección y segmentación

Para analizar una imagen:

```bash
python detect.py --imagen prueba.jpg --modelo best.pt
```

El resultado incluirá las detecciones y las máscaras de segmentación generadas por el modelo.

## 10. Resultados y limitaciones

El modelo fue entrenado y evaluado en Google Colab utilizando un conjunto de datos etiquetado con las clases bovino y persona.

Los resultados, las métricas de evaluación y el análisis de errores se documentan en el informe académico del proyecto.

El rendimiento obtenido corresponde al conjunto de imágenes utilizado y no garantiza el mismo desempeño en condiciones reales de vigilancia ganadera.

## 11. Trabajo futuro

- Ampliar la cantidad y diversidad de imágenes.
- Mejorar la detección de bovinos pequeños o parcialmente ocultos.
- Analizar el rendimiento en condiciones reales.
- Integrar el modelo en un sistema de vigilancia ganadera.

## 12. Autoría

Proyecto académico de Sistemas de Percepción en Robótica.
