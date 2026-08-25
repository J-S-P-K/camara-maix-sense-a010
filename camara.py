import serial
import time
import cv2
import numpy as np
import threading
import glob
from http.server import BaseHTTPRequestHandler, HTTPServer

BAUD = 115200
HEADER = b'\x00\xFF'
INTERVALO_FOTO = 1.0  # 1 foto cada 1.0 segundo
JPEG_QUALITY = 40     # Calidad de imagen baja para menor uso de CPU
STREAM_SIZE = (320, 320)

latest_frame = None
frame_lock = threading.Lock()

def conectar_camara():
    """Busca y conecta la cámara USB"""
    while True:
        puertos = sorted(glob.glob('/dev/ttyUSB*'))
        if puertos:
            puerto = puertos[0]
            try:
                ser = serial.Serial(puerto, BAUD, timeout=1)
                ser.reset_input_buffer()
                # Comandos de inicialización AT
                for cmd in [b"AT+ISP=1\r\n", b"AT+DISP=2\r\n", b"AT+UNIT=2\r\n", b"AT+BINN=1\r\n"]:
                    ser.write(cmd)
                    time.sleep(0.1)
                ser.reset_input_buffer()
                print(f"[✅] Cámara lista en {puerto}")
                return ser
            except Exception as e:
                print(f"[⚠️] Error al conectar en {puerto}: {e}")
        else:
            print("[🔍] Buscando cámara USB...", end="\r")
        time.sleep(2)

def camara_reader():
    global latest_frame
    ser = conectar_camara()
    buffer = bytearray()
    last_capture = 0
    fotos_count = 0

    while True:
        try:
            if ser.in_waiting > 0:
                buffer += ser.read(ser.in_waiting)
                
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
                    
                    # Solo procesamos la imagen si ya pasó 1 segundo desde la última
                    now = time.time()
                    if now - last_capture >= INTERVALO_FOTO:
                        last_capture = now
                        
                        cols = packet[14] if len(packet) > 14 and packet[14] > 0 else 100
                        rows = packet[15] if len(packet) > 15 and packet[15] > 0 else 100
                        img_start = 20
                        img_end = img_start + (rows * cols)
                        
                        if len(packet) >= img_end:
                            img_bytes = packet[img_start:img_end]
                            raw_img = np.array(img_bytes, dtype=np.uint8).reshape((rows, cols))
                            raw_img = cv2.flip(raw_img, 1)
                            
                            big_img = cv2.resize(raw_img, STREAM_SIZE, interpolation=cv2.INTER_NEAREST)
                            color_img = cv2.applyColorMap(big_img, cv2.COLORMAP_JET)
                            
                            _, jpeg = cv2.imencode('.jpg', color_img, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                            
                            with frame_lock:
                                latest_frame = jpeg.tobytes()
                                
                            fotos_count += 1
                            print(f"[📷] Foto #{fotos_count} capturada y enviada", end="\r")

        except Exception as e:
            print(f"\n[❌] Desconexión: {e}. Reconectando...")
            try:
                ser.close()
            except:
                pass
            time.sleep(1)
            ser = conectar_camara()
            buffer = bytearray()
            
        time.sleep(0.01)

class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            html = f"""<html><body style="background:#111; color:#fff; text-align:center; font-family:sans-serif;">
                <h2>ToF Stream (1 foto/seg)</h2>
                <img src="/stream" width="{STREAM_SIZE[0]}" height="{STREAM_SIZE[1]}" />
            </body></html>"""
            self.wfile.write(html.encode('utf-8'))
            
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with frame_lock:
                        frame = latest_frame
                    
                    if frame is not None:
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
                    
                    time.sleep(INTERVALO_FOTO) # Envía 1 foto cada segundo
            except Exception:
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        return # Desactiva logs molestos de HTTP en consola

if __name__ == '__main__':
    threading.Thread(target=camara_reader, daemon=True).start()
    print("[🌍] Servidor activo en http://0.0.0.0:8085")
    try:
        HTTPServer(('0.0.0.0', 8085), StreamingHandler).serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Cerrando...")
