import serial
import time
import cv2
import numpy as np
import threading
import glob
from http.server import BaseHTTPRequestHandler, HTTPServer

BAUD = 115200
HEADER = b'\x00\xFF'

latest_frame = None
frame_lock = threading.Lock()

def conectar_camara():
    """Busca en todos los puertos ttyUSB y verifica cuál transmite datos ToF reales"""
    while True:
        puertos = sorted(glob.glob('/dev/ttyUSB*'))
        if not puertos:
            print("[🔍] Buscando puertos USB disponibles...", end="\r")
            time.sleep(2)
            continue

        print(f"\n[Ports] Puertos detectados: {puertos}")

        for puerto in puertos:
            try:
                print(f"[🔌] Probando interfaz en {puerto}...")
                ser = serial.Serial(puerto, BAUD, timeout=1)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                
                # Comandos de activación de transmisión USB
                comandos = [b"AT+ISP=1\r\n", b"AT+DISP=2\r\n", b"AT+UNIT=2\r\n", b"AT+BINN=1\r\n"]
                for cmd in comandos:
                    ser.write(cmd)
                    time.sleep(0.1)
                
                ser.reset_input_buffer()
                
                # Leemos los primeros bytes para validar que sea la interfaz correcta
                time.sleep(0.3)
                if ser.in_waiting > 0:
                    sample = ser.read(min(ser.in_waiting, 500))
                    if HEADER in sample:
                        print(f"[✅] ¡ÉXITO! Se detectó flujo ToF válido en {puerto}")
                        return ser
                    else:
                        print(f"[⚠️] {puerto} está abierto pero no responde cabeceras ToF (probablemente puerto Debug).")
                else:
                    print(f"[⚠️] {puerto} no devolvió bytes. Probando siguiente...")
                
                ser.close()

            except PermissionError:
                print(f"[❌] Permiso denegado en {puerto}. Ejecutá: sudo chmod 666 {puerto}")
            except Exception as e:
                print(f"[⚠️] Error al probar {puerto}: {e}")
        
        print("[🔄] Reintentando búsqueda de interfaz ToF en 3 segundos...")
        time.sleep(3)

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
                    
                    cols = packet[14] if len(packet) > 14 and packet[14] > 0 else 100
                    rows = packet[15] if len(packet) > 15 and packet[15] > 0 else 100
                    
                    img_start = 20
                    img_end = img_start + (rows * cols)
                    
                    if len(packet) >= img_end:
                        img_bytes = packet[img_start:img_end]
                        raw_img = np.array(img_bytes, dtype=np.uint8).reshape((rows, cols))
                        
                        # VALIDACIÓN DE DATOS VALIDOS
                        val_promedio = np.mean(raw_img)
                        if val_promedio == 255.0:
                            # 255 significa "fuera de rango" o "sin datos" en el sensor
                            status_msg = " [⚠️ Trama en blanco/255] "
                        else:
                            status_msg = f" [OK - Dist. Promedio: {int(val_promedio)}mm] "

                        raw_img = cv2.flip(raw_img, 1) # Espejo
                        big_img = cv2.resize(raw_img, (500, 500), interpolation=cv2.INTER_NEAREST)
                        color_img = cv2.applyColorMap(big_img, cv2.COLORMAP_JET)
                        
                        _, jpeg = cv2.imencode('.jpg', color_img)
                        
                        with frame_lock:
                            latest_frame = jpeg.tobytes()
                            
                        frames_count += 1
                        print(f"📷 Frame #{frames_count} ({cols}x{rows}px){status_msg}| Bytes: {total_bytes}  ", end="\r")

        except Exception as e:
            print(f"\n\n[❌] Desconexión o error de lectura: {e}")
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(1)
            ser = conectar_camara()
            buffer = bytearray()
            
        time.sleep(0.001)

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = """
            <!DOCTYPE html>
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
    print("\n[🌍] Servidor Web Activo en el puerto 8085")
    server.serve_forever()

if __name__ == '__main__':
    thr = threading.Thread(target=camara_reader, daemon=True)
    thr.start()
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Cerrando...")
