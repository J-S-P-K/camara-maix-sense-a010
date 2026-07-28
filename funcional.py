import serial
import time
import cv2
import numpy as np
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
PORT   = "/dev/ttyUSB2"   # cambiá a USB1 si no funciona
BAUD   = 115200
HEADER = b'\x00\xFF'

latest_frame = None
frame_lock   = threading.Lock()


# ──────────────────────────────────────────────────────────────────────────────
# HILO 1: LECTOR DE LA CÁMARA TOF
# ──────────────────────────────────────────────────────────────────────────────
def camara_reader():
    global latest_frame

    # Rescate: si quedó en baud alto de sesión anterior, lo reseteamos
    for rescue_baud in [3000000, 921600, 460800]:
        try:
            s = serial.Serial(PORT, rescue_baud, timeout=0.2)
            s.write(b"AT+BAUD=2\r\n")   # volver a 115200
            time.sleep(0.2)
            s.close()
            break
        except Exception:
            pass

    print(f"[Serial] Conectando a {PORT} @ {BAUD} baud...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        ser.reset_input_buffer()
    except Exception as e:
        print(f"❌ [Serial] No se pudo abrir el puerto: {e}")
        return

    # Inicialización de comandos AT
    cmds = [
        b"AT+ISP=1\r\n",   # activar ISP
        b"AT+DISP=2\r\n",  # stream por USB (sin LCD)
        b"AT+UNIT=2\r\n",  # unidad = mm
        b"AT+BINN=1\r\n",  # resolución 100x100
        b"AT+FPS=10\r\n",  # 10 fps
    ]
    for cmd in cmds:
        ser.write(cmd)
        time.sleep(0.1)
    ser.reset_input_buffer()

    print("[Serial] Buscando frames... (esperá unos segundos)")

    buf       = bytearray()
    frame_cnt = 0

    while True:
        # Leemos todo lo disponible
        waiting = ser.in_waiting
        if waiting > 0:
            buf += ser.read(waiting)

        # Procesamos los paquetes completos
        while True:
            # 1. Buscar header
            idx = buf.find(HEADER)
            if idx == -1:
                buf = buf[-1:]
                break
            if idx > 0:
                buf = buf[idx:]   # descartar basura previa

            # 2. Necesitamos al menos los 4 bytes de cabecera
            if len(buf) < 4:
                break

            # 3. Leer longitud del payload
            payload_len = int.from_bytes(buf[2:4], 'little')
            packet_size = 4 + payload_len

            # Sanity check
            if packet_size < 100 or packet_size > 20000:
                buf = buf[2:]   # saltamos este header falso
                continue

            # 4. Esperar a tener el paquete completo
            if len(buf) < packet_size:
                break

            # 5. Extraer paquete
            packet = bytes(buf[:packet_size])
            buf    = buf[packet_size:]

            # 6. Parsear
# 6. Parsear
            try:
                # Fijamos la resolución exacta nativa de la Maix A010 en BINN=1
                cols = 100
                rows = 100
                
                img_start = 20  # 4 header/len + 16 meta
                img_end   = img_start + (rows * cols)  # 10,020 bytes total

                # Si el paquete que llegó es más chico que el tamaño del frame completo,
                # esperamos a que se complete en el buffer
                if len(packet) < img_end:
                    continue   

                img_bytes = packet[img_start:img_end]
                
                # Reconstruimos la imagen de 100x100
                raw_img = np.frombuffer(img_bytes, dtype=np.uint8).reshape((rows, cols))
                
                # Si la imagen se sigue viendo con franjas diagonales como la foto anterior,
                # descomenta la siguiente línea para transponer la matriz:
                # raw_img = raw_img.T

                raw_img = cv2.flip(raw_img, 1)   # Espejo horizontal

                # Escalar para visualización web (500x500)
                big_img   = cv2.resize(raw_img, (500, 500), interpolation=cv2.INTER_NEAREST)
                color_img = cv2.applyColorMap(big_img, cv2.COLORMAP_JET)

                _, jpeg = cv2.imencode('.jpg', color_img, [cv2.IMWRITE_JPEG_QUALITY, 85])

                with frame_lock:
                    latest_frame = jpeg.tobytes()

                frame_cnt += 1
                print(f"📷 Frame #{frame_cnt} | res={cols}x{rows} | payload_len={payload_len}   ", end="\r")

            except Exception as e:
                pass   # paquete corrupto o incompleto
        time.sleep(0.001)


# ──────────────────────────────────────────────────────────────────────────────
# HILO 2: SERVIDOR WEB MJPEG
# ──────────────────────────────────────────────────────────────────────────────
class StreamingHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Maix Sense A010</title>
  <style>
    body {
      background: #1e1e2e;
      color: #cdd6f4;
      font-family: sans-serif;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      height: 100vh;
      margin: 0;
    }
    h2 { margin-bottom: 16px; }
    .frame-box {
      border: 4px solid #45475a;
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    p { color: #a6adc8; margin-top: 12px; font-size: 14px; }
  </style>
</head>
<body>
  <h2>🎥 Maix Sense A010 — Mapa de Profundidad</h2>
  <div class="frame-box">
    <img src="/stream" width="500" height="500" />
  </div>
  <p>Azul = lejos &nbsp;|&nbsp; Rojo = cerca &nbsp;|&nbsp; Puerto: """ + PORT + """</p>
</body>
</html>"""
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
                    if frame:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.033)   # ~30 fps máx en el browser
            except Exception:
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return   # silenciar logs del servidor HTTP


def run_server():
    server = HTTPServer(('0.0.0.0', 8085), StreamingHandler)
    print("\n🌍 [Web] Servidor listo!")
    print("🌍 [Web] Abrí tu navegador en: http://localhost:8085\n")
    server.serve_forever()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    thr = threading.Thread(target=camara_reader, daemon=True)
    thr.start()
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Cerrando...")
