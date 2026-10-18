
# Sistema de Visión Artificial Ganadera

## 1. Descripción del proyecto

Este proyecto implementa un sistema de visión artificial para la detección y segmentación de bovinos y personas mediante YOLOv8n-seg.

El sistema identifica objetos presentes en imágenes, delimita sus regiones mediante máscaras de segmentación y muestra el nivel de confianza de cada predicción.

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
| 0 | cow (bovino) |
| 1 | person (persona) |

## 5. Estructura del proyecto

- `train.py`: script de entrenamiento.
- `val.py`: script de evaluación.
- `detect.py`: script de detección y segmentación.
- `requirements.txt`: dependencias de Python.
- `data.yaml`: configuración del conjunto de datos.
- `best.pt`: pesos del mejor modelo obtenido durante el entrenamiento.

El conjunto de datos debe estar organizado en las carpetas de entrenamiento, validación y prueba, con sus respectivas imágenes y etiquetas. Las rutas deben coincidir con la configuración de `data.yaml`.

## 6. Instalación

Se recomienda utilizar un entorno virtual de Python compatible con las versiones de las dependencias del proyecto.

Instalar las dependencias con:

```bash
pip install -r requirements.txt
```

## 7. Entrenamiento

Con el conjunto de datos disponible y `data.yaml` correctamente configurado, ejecutar:

```bash
python train.py
```

## 8. Evaluación

Con el conjunto de datos de evaluación y los pesos del modelo disponibles, ejecutar:

```bash
python val.py
```

## 9. Detección y segmentación

El script `detect.py` permite realizar inferencias sobre imágenes nuevas. Los argumentos y las rutas de entrada deben corresponder a los definidos en el código.

El resultado permite visualizar las detecciones, sus niveles de confianza y las máscaras de segmentación generadas por el modelo.

## 10. Resultados y limitaciones

El modelo fue entrenado y evaluado en Google Colab utilizando un conjunto de datos anotado con las clases bovino y persona.

Las métricas obtenidas y los ejemplos de predicción se presentan en las evidencias y el informe académico del proyecto.

El rendimiento está condicionado por el tamaño y la diversidad del conjunto de datos. Por tanto, los resultados experimentales no garantizan el mismo desempeño en condiciones reales de vigilancia ganadera.

## 11. Trabajo futuro

- Ampliar la cantidad y diversidad de imágenes.
- Mejorar la detección de bovinos pequeños o parcialmente ocultos.
- Evaluar el modelo en condiciones reales de campo.
- Integrar el modelo en un sistema de vigilancia ganadera.

## 12. Autoría

Proyecto académico de Sistemas de Percepción en Robótica.
