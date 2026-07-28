import serial
import time
import cv2
import numpy as np
import threading
import glob
from http.server import BaseHTTPRequestHandler, HTTPServer

BAUD = 115200
HEADER = b'\x00\xFF'

# --- CONFIGURACIÓN DE RENDIMIENTO Y CALIDAD ---
TARGET_FPS = 15                 # Tasa de cuadros por segundo deseada
FRAME_INTERVAL = 1.0 / TARGET_FPS # Intervalo de tiempo entre frames (~0.066s)
JPEG_QUALITY = 40               # Calidad de compresión JPEG (1 a 100, menor = menos peso/calidad)
STREAM_SIZE = (320, 320)        # Resolución visual (reducida de 500x500 para menor peso)

latest_frame = None
frame_lock = threading.Lock()

def encontrar_puerto_usb():
    """Busca cualquier puerto ttyUSB disponible (/dev/ttyUSB0, /dev/ttyUSB1, etc.)"""
    puertos = glob.glob('/dev/ttyUSB*')
    if puertos:
        return sorted(puertos)[0]
    return None

def conectar_camara():
    """Intenta abrir la cámara y reconfigurarla con comandos AT"""
    while True:
        puerto = encontrar_puerto_usb()
        if puerto:
            try:
                print(f"\n[🔌] Intentando conectar en {puerto}...")
                ser = serial.Serial(puerto, BAUD, timeout=1)
                ser.reset_input_buffer()
                ser.reset_output_buffer()
                
                # Enviar comandos de inicialización
                comandos = [b"AT+ISP=1\r\n", b"AT+DISP=2\r\n", b"AT+UNIT=2\r\n", b"AT+BINN=1\r\n"]
                for cmd in comandos:
                    ser.write(cmd)
                    time.sleep(0.1)
                
                ser.reset_input_buffer()
                print(f"[✅] ¡Conectado con éxito en {puerto}!")
                return ser
            except Exception as e:
                print(f"[⚠️] No se pudo abrir {puerto}: {e}")
        else:
            print("[🔍] Buscando cámara USB... (asegúrate de que esté enchufada)", end="\r")
        
        time.sleep(2)

def camara_reader():
    global latest_frame
    
    ser = conectar_camara()
    
    buffer = bytearray()
    total_bytes = 0
    frames_count = 0
    last_process_time = 0.0

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
                    
                    # Control de FPS: Descartar el procesamiento si no ha transcurrido el tiempo necesario
                    now = time.time()
                    if (now - last_process_time) < FRAME_INTERVAL:
                        continue
                    last_process_time = now

                    cols = packet[14] if len(packet) > 14 and packet[14] > 0 else 100
                    rows = packet[15] if len(packet) > 15 and packet[15] > 0 else 100
                    
                    img_start = 20
                    img_end = img_start + (rows * cols)
                    
                    if len(packet) >= img_end:
                        img_bytes = packet[img_start:img_end]
                        raw_img = np.array(img_bytes, dtype=np.uint8).reshape((rows, cols))
                        raw_img = cv2.flip(raw_img, 1)
                        
                        # 1. Reducción de resolución visual
                        big_img = cv2.resize(raw_img, STREAM_SIZE, interpolation=cv2.INTER_NEAREST)
                        color_img = cv2.applyColorMap(big_img, cv2.COLORMAP_JET)
                        
                        # 2. Reducción de Calidad JPEG (Parámetro IMWRITE_JPEG_QUALITY)
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                        _, jpeg = cv2.imencode('.jpg', color_img, encode_param)
                        
                        with frame_lock:
                            latest_frame = jpeg.tobytes()
                            
                        frames_count += 1
                        print(f"📷 [STREAM ACTIVO] Frame #{frames_count} ({cols}x{rows}px) | FPS: ~15 | Bytes: {total_bytes}  ", end="\r")

        except Exception as e:
            print(f"\n\n[❌] Desconexión detectada o Error I/O: {e}")
            print("[🔄] Reiniciando puerto y reconectando...")
            try:
                ser.close()
            except:
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
            html = f"""
            <html>
            <head><title>Radxa ToF Stream</title></head>
            <body style="background:#1e1e2e; color:#cdd6f4; font-family:sans-serif; display:flex; flex-direction:column; align-items:center; justify-content:center; height:100vh; margin:0;">
                <h2>Stream ToF (Sipeed A010) - 15 FPS</h2>
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
                    
                    # 3. Dormir el tiempo equivalente a 15 FPS en la transmisión HTTP (~0.066s)
                    time.sleep(FRAME_INTERVAL)
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
