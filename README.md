# Cámara ToF Sipeed MaixSense A010 + IA

Scripts para leer la cámara de profundidad ToF Sipeed A010 por USB desde una Radxa y verla en el navegador (`camara.py`, `aceleradaCamara.py`, `funcional.py`).

Todos los scripts usan la misma lectura de cámara que `camara.py` (puerto `/dev/ttyUSB0`, 115200 baudios, comandos AT) y el mismo servidor web en el puerto `8085`. Correr **uno solo a la vez**: comparten la cámara y el puerto.

- `camara.py`: visor básico.
- `aceleradaCamara.py`: igual, procesando la imagen en la GPU (OpenCL). Si la GPU falla, sigue en CPU.
- `funcional.py`: igual, pero si `/dev/ttyUSB0` no existe busca otro `/dev/ttyUSB*` y se reconecta solo si se desenchufa la cámara.

### Probar sin la cámara

`tests/camara_falsa.py` simula la cámara (mismo formato de paquetes) y ejecuta el script que le pases:

```bash
python tests/camara_falsa.py funcional.py
python tests/camara_falsa.py alertaAlmacenamiento.py
```

## Alerta de almacenamiento lleno con IA

Una red neuronal convolucional (CNN, TensorFlow → TensorFlow Lite) mira la imagen de profundidad de la zona del almacenamiento de la maqueta y la clasifica como **vacío** o **lleno**. Si la IA dice "lleno" con una confianza mayor a `UMBRAL_CONFIANZA` durante `TIEMPO_ALERTA` segundos seguidos (20 por defecto), se imprime una alerta en la terminal.

| Archivo | Qué hace | Dónde se corre |
|---|---|---|
| `tof_camara.py` | Lectura de la cámara, `ZONA` del almacenamiento y preprocesamiento compartido | — |
| `capturarDataset.py` | Guarda imágenes etiquetadas en `dataset/vacio` y `dataset/lleno` | Radxa |
| `entrenarModelo.py` | Entrena la CNN y genera `modelo_almacenamiento.tflite` | PC / Colab |
| `alertaAlmacenamiento.py` | Inferencia con TFLite, alerta por terminal y stream en `:8085` | Radxa |

### Pasos

1. En la Radxa: `pip install -r requirements.txt`
2. Ajustá `ZONA` en `tof_camara.py` para que encierre solo el almacenamiento.
3. Capturá datos (cuantas más situaciones distintas, mejor):
   ```bash
   python capturarDataset.py vacio
   python capturarDataset.py lleno
   ```
4. Copiá la carpeta `dataset/` a una PC (o Colab) y entrená:
   ```bash
   pip install -r requirements-entrenamiento.txt
   python entrenarModelo.py
   ```
5. Copiá `modelo_almacenamiento.tflite` a la Radxa y ejecutá:
   ```bash
   python alertaAlmacenamiento.py
   ```
   Abrí `http://<IP-DE-LA-RADXA>:8085` para ver la zona (rectángulo verde) y la predicción de la IA.
   Si todavía no está el `.tflite`, el stream se muestra igual con el texto "SIN MODELO".
