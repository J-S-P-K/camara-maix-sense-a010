"""Guarda imágenes de la cámara ToF etiquetadas para entrenar el modelo.

Uso:
    python capturarDataset.py vacio
    python capturarDataset.py lleno --cantidad 400 --intervalo 0.2

Mientras captura, mové/cambiá los objetos de la maqueta para que el modelo
vea muchas situaciones distintas de cada clase.
"""
import argparse
import os
import time

import cv2

from tof_camara import leer_frames

CLASES = ["vacio", "lleno"]


def main():
    parser = argparse.ArgumentParser(description="Captura de dataset para la IA del almacenamiento")
    parser.add_argument("etiqueta", choices=CLASES)
    parser.add_argument("--cantidad", type=int, default=300, help="imágenes a guardar")
    parser.add_argument("--intervalo", type=float, default=0.2, help="segundos entre capturas")
    parser.add_argument("--carpeta", default="dataset")
    args = parser.parse_args()

    destino = os.path.join(args.carpeta, args.etiqueta)
    os.makedirs(destino, exist_ok=True)

    for i in range(3, 0, -1):
        print(f"Preparate, la captura de '{args.etiqueta}' empieza en {i}...")
        time.sleep(1)

    guardadas = 0
    ultima = 0.0
    for img in leer_frames():
        ahora = time.monotonic()
        if ahora - ultima < args.intervalo:
            continue
        ultima = ahora
        nombre = os.path.join(destino, f"{int(time.time() * 1000)}.png")
        cv2.imwrite(nombre, img)
        guardadas += 1
        print(f"📸 {args.etiqueta}: {guardadas}/{args.cantidad}", end="\r")
        if guardadas >= args.cantidad:
            break

    print(f"\n[✅] Listo: {guardadas} imágenes guardadas en {destino}/")


if __name__ == "__main__":
    main()
