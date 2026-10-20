"""Entrena la red neuronal (CNN) que clasifica el almacenamiento en 'vacio' / 'lleno'
y la exporta a TensorFlow Lite.

Uso (conviene correrlo en una PC o en Google Colab):
    python entrenarModelo.py
    python entrenarModelo.py --epocas 50 --salida modelo_almacenamiento.tflite
"""
import argparse
import glob
import os

import cv2
import numpy as np
import tensorflow as tf

from capturarDataset import CLASES
from tof_camara import TAM_ENTRADA, preprocesar


def cargar_dataset(carpeta):
    imagenes, etiquetas = [], []
    for indice, clase in enumerate(CLASES):
        archivos = sorted(glob.glob(os.path.join(carpeta, clase, "*.png")))
        print(f"  {clase}: {len(archivos)} imágenes")
        for archivo in archivos:
            img = cv2.imread(archivo, cv2.IMREAD_GRAYSCALE)
            imagenes.append(preprocesar(img))
            etiquetas.append(indice)
    if len(set(etiquetas)) < len(CLASES):
        raise SystemExit("Faltan imágenes de alguna clase. Usá capturarDataset.py para ambas.")
    return np.array(imagenes), np.array(etiquetas, dtype=np.float32)


def aumentar(img, etiqueta):
    """Variaciones aleatorias para que el modelo generalice mejor."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.05)
    img = tf.clip_by_value(img + tf.random.normal(tf.shape(img), stddev=0.02), 0.0, 1.0)
    return img, etiqueta


def crear_modelo():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(TAM_ENTRADA, TAM_ENTRADA, 1)),
        tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),  # probabilidad de "lleno"
    ])


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento de la IA del almacenamiento")
    parser.add_argument("--carpeta", default="dataset")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--salida", default="modelo_almacenamiento.tflite")
    args = parser.parse_args()

    print("[1/4] Cargando dataset...")
    x, y = cargar_dataset(args.carpeta)

    orden = np.random.default_rng(42).permutation(len(x))
    x, y = x[orden], y[orden]
    corte = int(len(x) * 0.8)
    x_train, y_train, x_val, y_val = x[:corte], y[:corte], x[corte:], y[corte:]

    train = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
             .shuffle(len(x_train)).map(aumentar).batch(32))
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

    print("[2/4] Entrenando la red neuronal...")
    modelo = crear_modelo()
    modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    modelo.summary()
    modelo.fit(train, validation_data=val, epochs=args.epocas, callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    ])

    print("[3/4] Evaluando...")
    _, precision = modelo.evaluate(val, verbose=0)
    print(f"  Precisión en validación: {precision * 100:.1f}%")

    print("[4/4] Exportando a TensorFlow Lite...")
    conversor = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conversor.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(args.salida, "wb") as f:
        f.write(conversor.convert())
    print(f"[✅] Modelo guardado en {args.salida}")


if __name__ == "__main__":
    main()
