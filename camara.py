import serial
import time
import cv2
import numpy as np
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
PORT = "/dev/ttyUSB0"
BAUD = 115200  # Velocidad 100% segura para ARM/Radxa
HEADER = b'\x00\xFF'

latest_frame = None
frame_lock = threading.Lock()

# ──────────────────────────────────────────────────────────────────────────────
# HILO 1: LECTOR SERIAL Y DECODIFICADOR TOF
# ──────────────────────────────────────────────────────────────────────────────
def camara_reader():
    global latest_frame

    print(f"\n[1/4] Abriendo puerto {PORT} a {BAUD} baudios (Modo Seguro)...")
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        print("[✅] Puerto abierto correctamente. Sin errores de I/O.")
    except Exception as e:
        print(f"[❌] Error fatal al abrir el puerto: {e}")
        return

    print("[2/4] Configurando la cámara (Comandos AT)...")
    # AT+DISP=2 fuerza a la cámara a mandar el video SÓLO por USB para no saturarla
    comandos = [b"AT+ISP=1\r\n", b"AT+DISP=2\r\n", b"AT+UNIT=2\r\n", b"AT+BINN=1\r\n"]
    
    for cmd in comandos:
        ser.write(cmd)
        time.sleep(0.1)

    ser.reset_input_buffer()
    print("[✅] Comandos enviados.")
    print("[3/4] Escuchando flujo de video...")
    
    buffer = bytearray()
    total_bytes = 0
    last_debug_time = time.time()
    frames_decodificados = 0

    while True:
        try:
            waiting = ser.in_waiting
            if waiting > 0:
                chunk = ser.read(waiting)
                buffer += chunk
                total_bytes += len(chunk)
                
                # Búsqueda y decodificación del paquete binario
                while len(buffer) >= 4:
                    idx = buffer.find(HEADER)
                    
                    if idx == -1:
                        # Si no hay cabecera, descartamos basura reteniendo el último byte por si es la mitad del header
                        buffer = buffer[-1:]
                        break
                        
                    if idx > 0:
                        # Cortamos la basura antes del header
                        buffer = buffer[idx:]
                        if len(buffer) < 4:
                            break
                    
                    # Leemos el tamaño del payload (bytes 2 y 3 en little endian)
                    payload_len = int.from_bytes(buffer[2:4], 'little')
                    packet_size = 4 + payload_len
                    
                    # Filtro de seguridad: Un frame de 100x100 pesa aprox 10.000 bytes
                    if packet_size > 11000 or packet_size < 20:
                        buffer = buffer[2:] # Header corrupto, lo saltamos
                        continue
                    
                    if len(buffer) < packet_size:
                        break # Faltan llegar bytes para completar el frame
                    
                    # ¡Tenemos un paquete completo!
                    packet = buffer[:packet_size]
                    buffer = buffer[packet_size:]
                    
                    meta = packet[4:20]
                    cols = meta[0] if meta[0] > 0 else 100
                    rows = meta[1] if meta[1] > 0 else 100
                    
                    img_start = 20
                    img_end = img_start + (rows * cols)
                    
                    if len(packet) >= img_end:
                        img_bytes = packet[img_start:img_end]
                        raw_img = np.array(img_bytes, dtype=np.uint8).reshape((rows, cols))
                        raw_img = cv2.flip(raw_img, 1) # Efecto espejo
                        
                        big_img = cv2.resize(raw_img, (500, 500), interpolation=cv2.INTER_NEAREST)
                        color_img = cv2.applyColorMap(big_img, cv2.COLORMAP_JET)
                        
                        _, jpeg = cv2.imencode('.jpg', color_img)
                        
                        with frame_lock:
                            latest_frame = jpeg.tobytes()
                            
                        frames_decodificados += 1
                        last_debug_time = time.time() # Reseteamos el timer de debug
                        print(f"📷 [OK] Frame {frames_decodificados} capturado! ({cols}x{rows}) | RX: {total_bytes} bytes", end="\r")

            # ──────────────────────────────────────────────────────────────────
            # SISTEMA DE DEBUG EN TIEMPO REAL (Se activa si no hay frames)
            # ──────────────────────────────────────────────────────────────────
            if time.time() - last_debug_time > 2.5:
                print(f"\n[⚠️ DEBUG] Sin frames de video en los últimos 2.5s.")
                print(f"    -> Total de bytes recibidos desde inicio: {total_bytes}")
                print(f"    -> Tamaño del buffer actual esperando procesar: {len(buffer)} bytes")
                
                if len(buffer) > 0:
                    # Mostramos los primeros 40 caracteres para ver si es texto de error o ruido
                    hex_data = buffer[:40].hex()
                    try:
                        text_data = buffer[:40].decode('ascii', errors='ignore').replace('\r', '').replace('\n', ' ')
                        print(f"    -> Datos crudos (Texto): '{text_data}'")
                    except:
                        pass
                    print(f"    -> Datos crudos (Hex):   {hex_data}...")
                else:
                    print("    -> El puerto está en completo silencio (Revisar conexión física o energía).")
                    
                last_debug_time = time.time()

        except Exception as e:
            print(f"\n[❌] Excepción en la lectura serial: {e}")
            time.sleep(0.5)
            
        time.sleep(0.001)

# ──────────────────────────────────────────────────────────────────────────────
# HILO 2: SERVIDOR WEB (No se toca, funcionaba perfecto)
# ──────────────────────────────────────────────────────────────────────────────
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <html>
            <head><title>Radxa ToF Stream</title></head>
            <body style="background:#1e1e2e; color:#cdd6f4; font-family:sans-serif; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; margin:0;">
                <h2>Stream ToF (Sipeed A010)</h2>
                <div style="border: 4px solid #45475a; border-radius: 12px; overflow: hidden;">
                    <img src="/stream" width="500" height="500" />
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
                    
                    time.sleep(0.033)
            except Exception:
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return

def run_server():
    server = HTTPServer(('0.0.0.0', 8085), StreamingHandler)
    print("\n[4/4] 🌍 ¡Servidor Web Activo!")
    print("      👉 Ingresá desde el navegador de tu PC a: http://<IP-DE-LA-RADXA>:8085\n")
    server.serve_forever()

if __name__ == '__main__':
    thr = threading.Thread(target=camara_reader, daemon=True)
    thr.start()
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Cerrando programa...")
