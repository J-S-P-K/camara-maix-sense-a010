import serial
import time
import cv2
import numpy as np
import threading
import glob
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

# Misma base que camara.py (el stream que funciona), con dos agregados:
#   1. Si /dev/ttyUSB0 no existe, busca otro /dev/ttyUSB*.
#   2. Si la cámara se desenchufa, espera y se vuelve a conectar sola.
# Todo lo demás (comandos AT, lectura de paquetes, servidor web) es idéntico a camara.py.

PORT = "/dev/ttyUSB0"
BAUD = 115200
HEADER = b'\x00\xFF'

# --- CONFIGURACIÓN DE RENDIMIENTO Y CALIDAD ---
JPEG_QUALITY = 40        # Calidad JPEG (1 a 100). Reducirla alivia la CPU y la red
STREAM_SIZE = (320, 320) # Resolución de la transmisión (320x320 en lugar de 500x500)

latest_frame = None
frame_lock = threading.Lock()

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
                print(f"\n[1/4] Abriendo puerto {puerto} a {BAUD} baudios...")
                ser = serial.Serial(puerto, BAUD, timeout=1)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                print("[✅] Conexión USB establecida.")

                print("[2/4] Enviando comandos de inicialización...")
                comandos = [b"AT+ISP=1\r\n", b"AT+DISP=2\r\n", b"AT+UNIT=2\r\n", b"AT+BINN=1\r\n"]
                for cmd in comandos:
                    ser.write(cmd)
                    time.sleep(0.1)

                ser.reset_input_buffer()
                print("[3/4] Procesando flujo de imagen ToF...")
                return ser
            except Exception as e:
                print(f"[❌] Error al abrir el puerto {puerto}: {e}")
        else:
            print("[🔍] Buscando cámara USB... (asegurate de que esté enchufada)", end="\r")
        time.sleep(2)

def camara_reader():
    global latest_frame

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
                        raw_img = cv2.flip(raw_img, 1) # Espejo
                        
                        big_img = cv2.resize(raw_img, STREAM_SIZE, interpolation=cv2.INTER_NEAREST)
                        color_img = cv2.applyColorMap(big_img, cv2.COLORMAP_JET)
                        
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        _, jpeg = cv2.imencode('.jpg', color_img, encode_param)
                        
                        with frame_lock:
                            latest_frame = jpeg.tobytes()
                            
                        frames_count += 1
                        print(f"📷 [STREAM ACTIVO] Frame #{frames_count} ({cols}x{rows}px) | Total: {total_bytes} bytes  ", end="\r")

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

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""
            <html>
            <head><title>Radxa ToF Stream</title></head>
            <body style="background:#1e1e2e; color:#cdd6f4; font-family:sans-serif; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; margin:0;">
                <h2>Stream ToF (Sipeed A010)</h2>
                <div style="border: 4px solid #45475a; border-radius: 12px; overflow: hidden;">
                    <img src="/stream" width="{STREAM_SIZE[0]}" height="{STREAM_SIZE[1]}" />
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    
                    if frame is not None:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    
                    time.sleep(0.125)
            except Exception:
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return

def run_server():
    server = HTTPServer(('0.0.0.0', 8085), StreamingHandler)
    print("\n[4/4] 🌍 Servidor Web Activo!")
    print("      👉 Ingresá en tu navegador a: http://<IP-DE-LA-RADXA>:8085\n")
    server.serve_forever()

if __name__ == '__main__':
    thr = threading.Thread(target=camara_reader, daemon=True)
    thr.start()
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Cerrando...")
