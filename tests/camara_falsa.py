"""Cámara ToF simulada para probar los scripts sin la Sipeed A010 enchufada.

Uso (desde la carpeta del repo):
    python tests/camara_falsa.py funcional.py
    python tests/camara_falsa.py aceleradaCamara.py
    python tests/camara_falsa.py alertaAlmacenamiento.py
    python tests/camara_falsa.py capturarDataset.py vacio --cantidad 5

Reemplaza serial.Serial por una clase que genera paquetes con el mismo formato
que manda la cámara (cabecera 00 FF, largo, filas/columnas en 14/15, imagen
desde el byte 20) y hace que /dev/ttyUSB0 "exista". Después ejecuta el script
indicado tal cual, así que el stream se ve en http://localhost:8085.
"""
import glob
import math
import os
import runpy
import sys
import threading
import time

import numpy as np
import serial

FPS = 15
ROWS = COLS = 100
PUERTO_FALSO = "/dev/ttyUSB0"


def armar_paquete(img, frame_id):
    cabecera = bytearray(16)
    cabecera[10] = ROWS
    cabecera[11] = COLS
    cabecera[12] = frame_id & 0xFF
    cabecera[13] = (frame_id >> 8) & 0xFF
    cuerpo = bytes(cabecera) + img.tobytes()
    checksum = sum(cuerpo) & 0xFF
    payload = cuerpo + bytes([checksum, 0xDD])
    return b"\x00\xFF" + len(payload).to_bytes(2, "little") + payload


def imagen_falsa(t):
    """Fondo con gradiente + un círculo que se mueve (simula un objeto en el almacenamiento)."""
    y, x = np.mgrid[0:ROWS, 0:COLS]
    fondo = (x * 1.5 + 30).astype(np.uint8)
    cx = 50 + 30 * math.cos(t)
    cy = 50 + 30 * math.sin(t * 0.7)
    circulo = (x - cx) ** 2 + (y - cy) ** 2 < 15 ** 2
    img = fondo.copy()
    img[circulo] = 220
    return img


class SerialFalso:
    def __init__(self, port, baudrate=115200, timeout=None, **kwargs):
        if not port.startswith("/dev/ttyUSB"):
            raise serial.SerialException(f"could not open port {port}")
        self.port = port
        self.baudrate = baudrate
        self.is_open = True
        self._pendiente = bytearray()
        self._lock = threading.Lock()
        self._frame_id = 0
        self._comandos = []
        threading.Thread(target=self._generar, daemon=True).start()
        print(f"[camara_falsa] Puerto {port} abierto a {baudrate} baudios (simulado)")

    def _generar(self):
        inicio = time.monotonic()
        while self.is_open:
            t = time.monotonic() - inicio
            paquete = armar_paquete(imagen_falsa(t), self._frame_id)
            self._frame_id += 1
            # Lo entregamos en trozos, como llega por el puerto serie real
            with self._lock:
                mitad = len(paquete) // 2
                self._pendiente += paquete[:mitad]
            time.sleep(0.5 / FPS)
            with self._lock:
                self._pendiente += paquete[mitad:]
            time.sleep(0.5 / FPS)

    @property
    def in_waiting(self):
        with self._lock:
            return len(self._pendiente)

    def read(self, n=1):
        with self._lock:
            datos = bytes(self._pendiente[:n])
            del self._pendiente[:n]
        return datos

    def write(self, datos):
        self._comandos.append(bytes(datos))
        print(f"[camara_falsa] Comando recibido: {bytes(datos)!r}")
        return len(datos)

    def reset_input_buffer(self):
        with self._lock:
            self._pendiente.clear()

    def reset_output_buffer(self):
        pass

    def close(self):
        self.is_open = False


def instalar():
    serial.Serial = SerialFalso
    glob_original = glob.glob
    exists_original = os.path.exists
    glob.glob = lambda patron, *a, **k: [PUERTO_FALSO] if "ttyUSB" in patron else glob_original(patron, *a, **k)
    os.path.exists = lambda ruta: True if str(ruta).startswith("/dev/ttyUSB") else exists_original(ruta)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)) or ".")
    instalar()
    runpy.run_path(script, run_name="__main__")
