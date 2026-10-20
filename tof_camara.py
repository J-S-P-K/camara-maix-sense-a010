import glob
import time

import cv2
import numpy as np
import serial

BAUD = 115200
HEADER = b'\x00\xFF'
COMANDOS_INIT = [b"AT+ISP=1\r\n", b"AT+DISP=2\r\n", b"AT+UNIT=2\r\n", b"AT+BINN=1\r\n"]

# Zona del almacenamiento dentro de la imagen de 100x100 (x1, y1, x2, y2).
# Ajustala mirando el stream de alertaAlmacenamiento.py (el rectángulo verde).
# Si la cambiás, volvé a entrenar el modelo.
ZONA = (0, 0, 100, 100)

# Tamaño de la imagen que entra a la red neuronal
TAM_ENTRADA = 64


def encontrar_puerto_usb():
    puertos = glob.glob('/dev/ttyUSB*')
    return sorted(puertos)[0] if puertos else None


def conectar_camara():
    """Busca la cámara, la abre y la configura. Reintenta hasta lograrlo."""
    while True:
        puerto = encontrar_puerto_usb()
        if puerto:
            try:
                print(f"\n[🔌] Intentando conectar en {puerto}...")
                ser = serial.Serial(puerto, BAUD, timeout=1)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                for cmd in COMANDOS_INIT:
                    ser.write(cmd)
                    time.sleep(0.1)
                ser.reset_input_buffer()
                print(f"[✅] ¡Conectado con éxito en {puerto}!")
                return ser
            except Exception as e:
                print(f"[⚠️] No se pudo abrir {puerto}: {e}")
        else:
            print("[🔍] Buscando cámara USB... (asegurate de que esté enchufada)", end="\r")
        time.sleep(2)


def extraer_frames(buffer):
    """Saca del buffer todos los paquetes completos y devuelve (imágenes, buffer_restante)."""
    frames = []
    while len(buffer) >= 4:
        idx = buffer.find(HEADER)
        if idx == -1:
            buffer = buffer[-1:]
            break
        if idx > 0:
            buffer = buffer[idx:]
            if len(buffer) < 4:
                break

        payload_len = int.from_bytes(buffer[2:4], 'little')
        packet_size = 4 + payload_len
        if packet_size > 11000 or packet_size < 20:
            buffer = buffer[2:]
            continue
        if len(buffer) < packet_size:
            break

        packet = buffer[:packet_size]
        buffer = buffer[packet_size:]

        cols = packet[14] if packet[14] > 0 else 100
        rows = packet[15] if packet[15] > 0 else 100
        img_end = 20 + rows * cols
        if len(packet) >= img_end:
            img = np.frombuffer(bytes(packet[20:img_end]), dtype=np.uint8).reshape((rows, cols))
            frames.append(cv2.flip(img, 1))
    return frames, buffer


def leer_frames():
    """Generador infinito de imágenes de profundidad (uint8). Se reconecta solo si se desenchufa."""
    ser = conectar_camara()
    buffer = bytearray()
    while True:
        try:
            waiting = ser.in_waiting
            if waiting > 0:
                buffer += ser.read(waiting)
                frames, buffer = extraer_frames(buffer)
                yield from frames
        except Exception as e:
            print(f"\n[❌] Desconexión detectada o error I/O: {e}. Reconectando...")
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(1)
            ser = conectar_camara()
            buffer = bytearray()
        time.sleep(0.001)


def recortar_zona(img, zona=ZONA):
    x1, y1, x2, y2 = zona
    return img[y1:y2, x1:x2]


def preprocesar(img, zona=ZONA):
    """Imagen de profundidad -> tensor (TAM_ENTRADA, TAM_ENTRADA, 1) float32 en [0, 1].
    Se usa igual al entrenar y al predecir."""
    recorte = recortar_zona(img, zona)
    recorte = cv2.resize(recorte, (TAM_ENTRADA, TAM_ENTRADA), interpolation=cv2.INTER_AREA)
    return (recorte.astype(np.float32) / 255.0)[..., np.newaxis]
