import serial
import time
import cv2
import numpy as np
import threading
import glob
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

# ──────────────────────────────────────────────────────────────────────────────
# ESTADO GLOBAL
# ──────────────────────────────────────────────────────────────────────────────
BAUD = 115200
HEADER = b'\x00\xFF'

latest_frame = None
frame_lock = threading.Lock()

# Configuración de puertos seleccionados por el usuario
main_port = None       # Puerto de Video / Datos ToF
secondary_port = None  # Puerto Secundario / Debug
port_change_requested = False
status_message = "Esperando selección de puerto..."

# ──────────────────────────────────────────────────────────────────────────────
# HILO DE LECTURA DE CÁMARA
# ──────────────────────────────────────────────────────────────────────────────
def camara_reader():
    global latest_frame, main_port, port_change_requested, status_message
    
    ser = None
    buffer = bytearray()
    total_bytes = 0
    frames_count = 0

    while True:
        # Si el usuario cambió el puerto desde la web o no hay conexión activa
        if port_change_requested or ser is None or not ser.is_open:
            if ser and ser.is_open:
                try:
                    ser.close()
                except Exception:
                    pass
            
            port_change_requested = False
            
            # Si no hay puerto elegido por el usuario, busca el primero disponible
            if not main_port:
                puertos = sorted(glob.glob('/dev/ttyUSB*'))
                if puertos:
                    main_port = puertos[-1] # Elige el último por defecto (ej: USB2)
                else:
                    status_message = "❌ Sin puertos USB detectados en el sistema."
                    time.sleep(2)
                    continue

            status_message = f"🔌 Intentando conectar a {main_port}..."
            print(f"\n[Puerto] Abriendo puerto asignado: {main_port}")
            
            try:
                ser = serial.Serial(main_port, BAUD, timeout=1.5)
                time.sleep(1.2) # Pausa de estabilización eléctrica
                
                ser.reset_input_buffer()
                ser.reset_output_buffer()

                # Mandamos comandos de inicialización
                for cmd in [b"AT+ISP=1\r\n", b"AT+DISP=2\r\n", b"AT+UNIT=2\r\n", b"AT+BINN=1\r\n"]:
                    ser.write(cmd)
                    time.sleep(0.1)

                ser.reset_input_buffer()
                buffer = bytearray()
                status_message = f"🟢 Conectado a {main_port} | Recibiendo datos..."
                print(f"[✅] Conexión establecida en {main_port}")

            except PermissionError:
                status_message = f"❌ Permiso denegado en {main_port}. Ejecutá: sudo chmod 666 {main_port}"
                print(f"[❌] Error de permisos en {main_port}")
                ser = None
                time.sleep(3)
                continue
            except Exception as e:
                status_message = f"⚠️ Error abriendo {main_port}: {e}"
                print(f"[⚠️] Fallo al abrir {main_port}: {e}")
                ser = None
                time.sleep(3)
                continue

        # Lectura de frames ToF
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
                        
                        raw_img = cv2.flip(raw_img, 1) # Espejo
                        big_img = cv2.resize(raw_img, (500, 500), interpolation=cv2.INTER_NEAREST)
                        color_img = cv2.applyColorMap(big_img, cv2.COLORMAP_JET)
                        
                        _, jpeg = cv2.imencode('.jpg', color_img)
                        
                        with frame_lock:
                            latest_frame = jpeg.tobytes()
                            
                        frames_count += 1
                        status_message = f"🟢 Transmitiendo ({main_port}) | Frame #{frames_count}"
                        print(f"📷 Stream activo en {main_port} | Frame #{frames_count}  ", end="\r")

        except Exception as e:
            print(f"\n[❌] Error de lectura o desconexión en {main_port}: {e}")
            status_message = f"⚠️ Desconexión en {main_port}. Reintentando..."
            ser = None
            time.sleep(2)

        time.sleep(0.001)

# ──────────────────────────────────────────────────────────────────────────────
# SERVIDOR WEB CON CONTROLES
# ──────────────────────────────────────────────────────────────────────────────
class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_frame, main_port, secondary_port, status_message
        
        if self.path == '/':
            puertos_disponibles = sorted(glob.glob('/dev/ttyUSB*'))
            
            # Generar opciones dinámicas para el select
            options_main = ""
            options_sec = ""
            
            for p in puertos_disponibles:
                selected_m = "selected" if p == main_port else ""
                selected_s = "selected" if p == secondary_port else ""
                options_main += f'<option value="{p}" {selected_m}>{p}</option>\n'
                options_sec += f'<option value="{p}" {selected_s}>{p}</option>\n'

            if not puertos_disponibles:
                options_main = '<option value="">Sin puertos USB</option>'
                options_sec = '<option value="">Sin puertos USB</option>'

            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Radxa ToF Control & Stream</title>
                <meta charset="utf-8">
                <style>
                    body {{
                        background: #1e1e2e;
                        color: #cdd6f4;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        min-height: 100vh;
                        margin: 0;
                        padding: 20px;
                    }}
                    .container {{
                        background: #181825;
                        padding: 25px;
                        border-radius: 16px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        gap: 20px;
                        max-width: 550px;
                        width: 100%;
                    }}
                    .stream-box {{
                        border: 3px solid #45475a;
                        border-radius: 12px;
                        overflow: hidden;
                        background: #11111b;
                    }}
                    .status-bar {{
                        background: #313244;
                        color: #a6adc8;
                        padding: 10px 15px;
                        border-radius: 8px;
                        width: 100%;
                        text-align: center;
                        font-size: 14px;
                        font-weight: bold;
                        box-sizing: border-box;
                    }}
                    form {{
                        display: flex;
                        flex-direction: column;
                        gap: 12px;
                        width: 100%;
                        background: #313244;
                        padding: 15px;
                        border-radius: 10px;
                        box-sizing: border-box;
                    }}
                    .form-group {{
                        display: flex;
                        justify-content: space-between;
                        align-items: center;
                    }}
                    label {{
                        font-size: 14px;
                        font-weight: 600;
                    }}
                    select {{
                        background: #1e1e2e;
                        color: #cdd6f4;
                        border: 1px solid #585b70;
                        padding: 8px 12px;
                        border-radius: 6px;
                        font-size: 14px;
                        outline: none;
                        cursor: pointer;
                    }}
                    button {{
                        background: #89b4fa;
                        color: #11111b;
                        border: none;
                        padding: 10px;
                        border-radius: 6px;
                        font-weight: bold;
                        font-size: 14px;
                        cursor: pointer;
                        transition: background 0.2s;
                        margin-top: 5px;
                    }}
                    button:hover {{
                        background: #b4befe;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2 style="margin: 0; color: #89b4fa;">Control ToF (Radxa Dragon)</h2>
                    
                    <div class="status-bar">
                        {status_message}
                    </div>

                    <div class="stream-box">
                        <img src="/stream" width="500" height="500" alt="Stream de video ToF" />
                    </div>

                    <form action="/set_ports" method="POST">
                        <div class="form-group">
                            <label for="main_port">Puerto Principal (Datos/Video):</label>
                            <select name="main_port" id="main_port">
                                {options_main}
                            </select>
                        </div>

                        <div class="form-group">
                            <label for="sec_port">Puerto Secundario (Debug):</label>
                            <select name="sec_port" id="sec_port">
                                {options_sec}
                            </select>
                        </div>

                        <button type="submit">🔄 Guardar y Reconectar Puerto</button>
                    </form>
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

    def do_POST(self):
        global main_port, secondary_port, port_change_requested
        
        if self.path == '/set_ports':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length).decode('utf-8')
            params = urllib.parse.parse_qs(post_data)
            
            new_main = params.get('main_port', [None])[0]
            new_sec = params.get('sec_port', [None])[0]
            
            if new_main:
                main_port = new_main
                secondary_port = new_sec
                port_change_requested = True
                print(f"\n[Web Control] Usuario asignó -> Principal: {main_port} | Secundario: {secondary_port}")
            
            # Redireccionar de vuelta a la página principal
            self.send_response(303)
            self.send_header('Location', '/')
            self.end_headers()

    def log_message(self, format, *args):
        return

def run_server():
    server = HTTPServer(('0.0.0.0', 8085), StreamingHandler)
    print("\n[🌍] Servidor Web de Control Activo en el puerto 8085")
    print("     👉 Ingresá en tu navegador a: http://<IP-DE-LA-RADXA>:8085\n")
    server.serve_forever()

if __name__ == '__main__':
    thr = threading.Thread(target=camara_reader, daemon=True)
    thr.start()
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n👋 Cerrando servidor...")
