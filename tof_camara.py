"""Lectura de la cámara ToF Sipeed A010 compartida por los scripts de IA.

Es el mismo código de lectura de camara.py (el stream que funciona), separado en
funciones. Las únicas diferencias: si /dev/ttyUSB0 no existe busca otro
/dev/ttyUSB*, y si la cámara se desenchufa se vuelve a conectar sola.
"""
import glob
import os
import time

import cv2
import numpy as np
import serial

PORT = "/dev/ttyUSB0"
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
    """Usa PORT si existe; si no, el primer /dev/ttyUSB* que encuentre."""
    if os.path.exists(PORT):
        return PORT
    puertos = sorted(glob.glob('/dev/ttyUSB*'))
    return puertos[0] if puertos else None


def conectar_camara():
    """Abre la cámara y manda los comandos AT igual que camara.py. Reintenta hasta lograrlo."""
    while True:
        puerto = encontrar_puerto_usb()
        if puerto:
            try:
                print(f"\n[1/3] Abriendo puerto {puerto} a {BAUD} baudios...")
                ser = serial.Serial(puerto, BAUD, timeout=1)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                print("[✅] Conexión USB establecida.")

                print("[2/3] Enviando comandos de inicialización...")
                for cmd in COMANDOS_INIT:
                    ser.write(cmd)
                    time.sleep(0.1)

                ser.reset_input_buffer()
                print("[3/3] Procesando flujo de imagen ToF...")
                return ser
            except Exception as e:
                print(f"[❌] Error al abrir el puerto {puerto}: {e}")
        else:
            print("[🔍] Buscando cámara USB... (asegurate de que esté enchufada)", end="\r")
        time.sleep(2)


def extraer_frames(buffer):
    """Saca del buffer todos los paquetes completos (misma lógica que camara.py)
    y devuelve (imágenes 100x100 ya espejadas, buffer_restante)."""
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

        # La resolución (100x100) está en los índices 14 y 15 del paquete
        cols = packet[14] if len(packet) > 14 and packet[14] > 0 else 100
        rows = packet[15] if len(packet) > 15 and packet[15] > 0 else 100

        img_start = 20
        img_end = img_start + (rows * cols)

        if len(packet) >= img_end:
            img_bytes = packet[img_start:img_end]
            raw_img = np.array(img_bytes, dtype=np.uint8).reshape((rows, cols))
            frames.append(cv2.flip(raw_img, 1))  # Espejo
    return frames, buffer


def leer_frames():
    """Generador infinito de imágenes de profundidad (uint8, 100x100).
    Mismo bucle de lectura que camara.py; además se reconecta si se desenchufa la cámara."""
    ser = conectar_camara()
    buffer = bytearray()
    total_bytes = 0
    frames_count = 0

    while True:
        try:
            waiting = ser.in_waiting
            if waiting > 0:
                chunk = ser.read(waiting)
                buffer += chunk
                total_bytes += len(chunk)

                frames, buffer = extraer_frames(buffer)
                for img in frames:
                    frames_count += 1
                    yield img

        except (serial.SerialException, OSError) as e:
            # Solo acá reconectamos: el puerto se cerró o la cámara se desenchufó
            print(f"\n[❌] Desconexión detectada: {e}")
            print("[🔄] Reconectando...")
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(1)
            ser = conectar_camara()
            buffer = bytearray()
        except Exception as e:
            # Cualquier otro error se ignora y se sigue leyendo, igual que camara.py
            print(f"\n[⚠️] Excepción en lectura serial: {e}")
            time.sleep(0.2)

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
